#include "ndtbl/ndtbl.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct parsed_text_table
{
  std::string field_name;
  std::vector<ndtbl::Axis> axes;
  std::vector<double> values;
};

std::string
read_file(const std::filesystem::path& path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open text table: " + path.string());
  }

  return std::string((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
}

std::string
strip_comments(std::string_view input)
{
  std::string output;
  output.reserve(input.size());

  bool in_line_comment = false;
  bool in_block_comment = false;
  bool in_string = false;

  for (std::size_t i = 0; i < input.size(); ++i) {
    const char c = input[i];
    const char next = i + 1 < input.size() ? input[i + 1] : '\0';

    if (in_line_comment) {
      if (c == '\n') {
        in_line_comment = false;
        output.push_back(c);
      }
      continue;
    }

    if (in_block_comment) {
      if (c == '*' && next == '/') {
        in_block_comment = false;
        ++i;
      }
      continue;
    }

    if (!in_string && c == '/' && next == '/') {
      in_line_comment = true;
      ++i;
      continue;
    }

    if (!in_string && c == '/' && next == '*') {
      in_block_comment = true;
      ++i;
      continue;
    }

    if (c == '"') {
      in_string = !in_string;
    }

    output.push_back(c);
  }

  return output;
}

class token_stream
{
public:
  explicit token_stream(std::string input)
    : storage_(std::move(input))
    , tokens_(tokenize(storage_))
    , index_(0)
  {
  }

  bool empty() const { return index_ >= tokens_.size(); }

  std::string_view peek(std::size_t offset = 0) const
  {
    if (index_ + offset >= tokens_.size()) {
      throw std::runtime_error("unexpected end of token stream");
    }
    return tokens_[index_ + offset];
  }

  std::string_view next()
  {
    const std::string_view token = peek();
    ++index_;
    return token;
  }

  void expect(std::string_view expected)
  {
    const std::string_view token = next();
    if (token != expected) {
      throw std::runtime_error("expected token '" + std::string(expected) +
                               "' but found '" + std::string(token) + "'");
    }
  }

private:
  static bool is_punctuation(char c)
  {
    return c == '{' || c == '}' || c == '(' || c == ')' || c == ';';
  }

  static std::vector<std::string_view> tokenize(std::string_view input)
  {
    std::vector<std::string_view> tokens;
    std::size_t i = 0;
    while (i < input.size()) {
      const char c = input[i];

      if (std::isspace(static_cast<unsigned char>(c))) {
        ++i;
        continue;
      }

      if (is_punctuation(c)) {
        tokens.push_back(input.substr(i, 1));
        ++i;
        continue;
      }

      if (c == '"') {
        std::size_t j = i + 1;
        while (j < input.size() && input[j] != '"') {
          ++j;
        }
        if (j >= input.size()) {
          throw std::runtime_error("unterminated quoted string in text table");
        }
        tokens.push_back(input.substr(i + 1, j - i - 1));
        i = j + 1;
        continue;
      }

      std::size_t j = i;
      while (j < input.size() &&
             !std::isspace(static_cast<unsigned char>(input[j])) &&
             !is_punctuation(input[j])) {
        ++j;
      }
      tokens.push_back(input.substr(i, j - i));
      i = j;
    }

    return tokens;
  }

  std::string storage_;
  std::vector<std::string_view> tokens_;
  std::size_t index_;
};

std::size_t
parse_size(std::string_view token)
{
  return static_cast<std::size_t>(std::stoull(std::string(token)));
}

double
parse_scalar(std::string_view token)
{
  return std::stod(std::string(token));
}

void
skip_parenthesized(token_stream& tokens)
{
  tokens.expect("(");
  int depth = 1;
  while (depth > 0) {
    const std::string_view token = tokens.next();
    if (token == "(") {
      ++depth;
    } else if (token == ")") {
      --depth;
    }
  }
}

void
skip_braced(token_stream& tokens)
{
  tokens.expect("{");
  int depth = 1;
  while (depth > 0) {
    const std::string_view token = tokens.next();
    if (token == "{") {
      ++depth;
    } else if (token == "}") {
      --depth;
    }
  }
}

std::vector<double>
parse_scalar_list(token_stream& tokens)
{
  const std::size_t count = parse_size(tokens.next());
  tokens.expect("(");
  std::vector<double> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = parse_scalar(tokens.next());
  }
  tokens.expect(")");
  return values;
}

void
parse_nested_values(token_stream& tokens,
                    std::size_t level,
                    std::vector<std::size_t>& extents,
                    std::vector<double>& values)
{
  const std::size_t extent = parse_size(tokens.next());
  if (extents.size() == level) {
    extents.push_back(extent);
  } else if (extents[level] != extent) {
    throw std::runtime_error("ragged OpenFOAM text table is not supported");
  }

  tokens.expect("(");
  const bool nested = tokens.peek(1) == "(";

  if (nested) {
    for (std::size_t i = 0; i < extent; ++i) {
      parse_nested_values(tokens, level + 1, extents, values);
    }
  } else {
    for (std::size_t i = 0; i < extent; ++i) {
      values.push_back(parse_scalar(tokens.next()));
    }
  }

  tokens.expect(")");
}

std::string
default_field_name(std::string_view table_name,
                   const std::filesystem::path& path)
{
  if (!table_name.empty()) {
    if (table_name.ends_with("_table")) {
      return std::string(table_name.substr(0, table_name.size() - 6));
    }
    return std::string(table_name);
  }

  const std::string filename = path.filename().string();
  if (std::string_view(filename).ends_with("_table")) {
    return filename.substr(0, filename.size() - 6);
  }
  return filename;
}

parsed_text_table
parse_text_table(const std::filesystem::path& path)
{
  token_stream tokens(strip_comments(read_file(path)));
  std::vector<double> axis_mins;
  std::vector<double> axis_maxs;
  std::vector<std::size_t> extents;
  std::vector<double> values;
  std::string table_name;

  while (!tokens.empty()) {
    const std::string_view key = tokens.next();

    if (key == "FoamFile") {
      skip_braced(tokens);
      continue;
    }

    if (key == "axisMins") {
      axis_mins = parse_scalar_list(tokens);
      tokens.expect(";");
      continue;
    }

    if (key == "axisMaxs") {
      axis_maxs = parse_scalar_list(tokens);
      tokens.expect(";");
      continue;
    }

    table_name = std::string(key);
    parse_nested_values(tokens, 0, extents, values);
    tokens.expect(";");
  }

  if (extents.empty()) {
    throw std::runtime_error("no nd values found in text table: " +
                             path.string());
  }

  if (axis_mins.empty()) {
    axis_mins.assign(extents.size(), 0.0);
  }

  if (axis_maxs.empty()) {
    axis_maxs.assign(extents.size(), 1.0);
  }

  if (axis_mins.size() != extents.size() ||
      axis_maxs.size() != extents.size()) {
    throw std::runtime_error(
      "axis metadata dimensionality does not match text table nesting");
  }

  parsed_text_table parsed;
  parsed.field_name = default_field_name(table_name, path);
  parsed.axes.reserve(extents.size());
  for (std::size_t axis = 0; axis < extents.size(); ++axis) {
    parsed.axes.push_back(
      ndtbl::Axis::uniform(axis_mins[axis], axis_maxs[axis], extents[axis]));
  }
  parsed.values = values;
  return parsed;
}

ndtbl::AnyFieldGroup
pack_group(std::span<const parsed_text_table> tables, bool use_float)
{
  if (tables.empty()) {
    throw std::runtime_error("no input tables supplied");
  }

  const std::vector<ndtbl::Axis>& reference_axes = tables.front().axes;
  const std::size_t point_count = tables.front().values.size();
  std::vector<std::string> field_names;
  field_names.reserve(tables.size());

  for (std::size_t i = 0; i < tables.size(); ++i) {
    if (tables[i].axes.size() != reference_axes.size()) {
      throw std::runtime_error("cannot pack tables with different dimensions");
    }

    for (std::size_t axis = 0; axis < reference_axes.size(); ++axis) {
      if (!reference_axes[axis].equivalent(tables[i].axes[axis])) {
        throw std::runtime_error("cannot pack tables with different grids");
      }
    }

    if (tables[i].values.size() != point_count) {
      throw std::runtime_error(
        "cannot pack tables with different point counts");
    }

    field_names.push_back(tables[i].field_name);
  }

  if (use_float) {
    std::vector<float> interleaved;
    interleaved.reserve(point_count * tables.size());
    for (std::size_t point = 0; point < point_count; ++point) {
      for (std::size_t field = 0; field < tables.size(); ++field) {
        interleaved.push_back(static_cast<float>(tables[field].values[point]));
      }
    }
    return ndtbl::detail::make_any_group<float>(
      reference_axes, field_names, interleaved);
  }

  std::vector<double> interleaved;
  interleaved.reserve(point_count * tables.size());
  for (std::size_t point = 0; point < point_count; ++point) {
    for (std::size_t field = 0; field < tables.size(); ++field) {
      interleaved.push_back(tables[field].values[point]);
    }
  }
  return ndtbl::detail::make_any_group<double>(
    reference_axes, field_names, interleaved);
}

void
print_usage()
{
  std::cerr << "Usage: ndtbl-convert [--precision float|double] output.ndtbl "
               "input_table...\n";
}

} // namespace

int
main(int argc, char** argv)
{
  try {
    bool use_float = false;
    int arg_index = 1;

    if (argc > 3 && std::string(argv[arg_index]) == "--precision") {
      const std::string precision = argv[arg_index + 1];
      if (precision == "float") {
        use_float = true;
      } else if (precision == "double") {
        use_float = false;
      } else {
        throw std::runtime_error("unsupported precision: " + precision);
      }
      arg_index += 2;
    }

    if (argc - arg_index < 2) {
      print_usage();
      return 1;
    }

    const std::filesystem::path output_path = argv[arg_index++];
    std::vector<parsed_text_table> tables;
    for (int i = arg_index; i < argc; ++i) {
      tables.push_back(parse_text_table(argv[i]));
    }

    ndtbl::write_group(output_path.string(), pack_group(tables, use_float));
    std::cout << "Wrote " << output_path.string() << " with " << tables.size()
              << " field(s)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ndtbl-convert: " << error.what() << '\n';
    return 1;
  }
}
