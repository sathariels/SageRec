#include "sagerec/movielens_100k.hpp"
#include "sagerec/error.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace sagerec {
namespace {

struct RawRow {
  std::int32_t source_user_id = 0;
  std::int32_t source_movie_id = 0;
  std::int32_t rating = 0;
  std::int64_t timestamp = 0;
};

bool is_blank(std::string_view line) {
  for (const char c : line) {
    if (c != ' ' && c != '\t') {
      return false;
    }
  }
  return true;
}

std::string_view trim_spaces(std::string_view field) {
  while (!field.empty() && field.front() == ' ') {
    field.remove_prefix(1);
  }
  while (!field.empty() && field.back() == ' ') {
    field.remove_suffix(1);
  }
  return field;
}

std::string format_field(std::string_view field) {
  return std::string(field);
}

template <typename T>
T parse_int_field(std::string_view raw_field,
                  std::size_t line_number,
                  const char* field_name) {
  const std::string_view field = trim_spaces(raw_field);
  T value{};
  const char* begin = field.data();
  const char* end = field.data() + field.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec == std::errc::result_out_of_range) {
    std::ostringstream out;
    out << "Malformed MovieLens 100K row " << line_number << ": " << field_name << " '"
        << format_field(field) << "' is out of range for its integer type";
    throw GraphError(out.str());
  }
  if (ec != std::errc{} || ptr != end) {
    std::ostringstream out;
    out << "Malformed MovieLens 100K row " << line_number << ": " << field_name << " '"
        << format_field(field) << "' is not an integer";
    throw GraphError(out.str());
  }
  return value;
}

RawRow parse_row(std::string_view line, std::size_t line_number) {
  std::string_view fields[4];
  std::size_t count = 0;
  std::size_t start = 0;
  while (true) {
    const std::size_t tab = line.find('\t', start);
    const std::string_view field =
        tab == std::string_view::npos ? line.substr(start) : line.substr(start, tab - start);
    if (count < 4) {
      fields[count] = field;
    }
    ++count;
    if (tab == std::string_view::npos) {
      break;
    }
    start = tab + 1;
  }

  if (count != 4) {
    std::ostringstream out;
    out << "Malformed MovieLens 100K row " << line_number
        << ": expected 4 tab-separated fields (user_id, movie_id, rating, timestamp), got "
        << count;
    throw GraphError(out.str());
  }

  RawRow row;
  row.source_user_id = parse_int_field<std::int32_t>(fields[0], line_number, "user_id");
  row.source_movie_id = parse_int_field<std::int32_t>(fields[1], line_number, "movie_id");
  row.rating = parse_int_field<std::int32_t>(fields[2], line_number, "rating");
  row.timestamp = parse_int_field<std::int64_t>(fields[3], line_number, "timestamp");

  if (row.source_user_id <= 0) {
    std::ostringstream out;
    out << "Malformed MovieLens 100K row " << line_number << ": user_id " << row.source_user_id
        << " is not a positive source ID";
    throw GraphError(out.str());
  }
  if (row.source_movie_id <= 0) {
    std::ostringstream out;
    out << "Malformed MovieLens 100K row " << line_number << ": movie_id " << row.source_movie_id
        << " is not a positive source ID";
    throw GraphError(out.str());
  }
  return row;
}

std::vector<std::int32_t> sorted_unique_ids(const std::vector<RawRow>& rows, bool users) {
  std::vector<std::int32_t> ids;
  ids.reserve(rows.size());
  for (const RawRow& row : rows) {
    ids.push_back(users ? row.source_user_id : row.source_movie_id);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.size() > static_cast<std::size_t>(std::numeric_limits<NodeId>::max())) {
    throw GraphError(
        users ? "unique MovieLens 100K user count exceeds NodeId range"
              : "unique MovieLens 100K movie count exceeds NodeId range");
  }
  return ids;
}

NodeId local_id(const std::vector<std::int32_t>& source_ids, std::int32_t source_id) {
  const auto it = std::lower_bound(source_ids.begin(), source_ids.end(), source_id);
  return static_cast<NodeId>(it - source_ids.begin());
}

}  // namespace

std::vector<std::pair<NodeId, NodeId>> MovieLens100kRatings::local_pairs() const {
  std::vector<std::pair<NodeId, NodeId>> pairs;
  pairs.reserve(interactions_.size());
  for (const MovieLens100kInteraction& interaction : interactions_) {
    pairs.emplace_back(interaction.user_id, interaction.movie_id);
  }
  return pairs;
}

MovieLens100kRatings parse_movielens_100k(std::string_view text) {
  std::vector<RawRow> rows;
  std::map<std::pair<std::int32_t, std::int32_t>, std::size_t> first_seen;

  std::size_t pos = 0;
  std::size_t line_number = 0;
  while (pos <= text.size()) {
    const std::size_t newline = text.find('\n', pos);
    std::string_view line = newline == std::string_view::npos
                                ? text.substr(pos)
                                : text.substr(pos, newline - pos);
    pos = newline == std::string_view::npos ? text.size() + 1 : newline + 1;
    ++line_number;

    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    if (is_blank(line)) {
      continue;
    }

    const RawRow row = parse_row(line, line_number);
    const auto key = std::make_pair(row.source_user_id, row.source_movie_id);
    const auto [it, inserted] = first_seen.emplace(key, line_number);
    if (!inserted) {
      std::ostringstream out;
      out << "Duplicate MovieLens 100K interaction at row " << line_number << ": source user_id="
          << row.source_user_id << " movie_id=" << row.source_movie_id << " already seen at row "
          << it->second;
      throw GraphError(out.str());
    }
    rows.push_back(row);
  }

  if (rows.empty()) {
    throw GraphError(
        "MovieLens 100K input is empty; expected at least one tab-separated u.data row");
  }

  MovieLens100kRatings parsed;
  parsed.user_source_ids_ = sorted_unique_ids(rows, true);
  parsed.movie_source_ids_ = sorted_unique_ids(rows, false);
  parsed.interactions_.reserve(rows.size());
  for (const RawRow& row : rows) {
    MovieLens100kInteraction interaction;
    interaction.user_id = local_id(parsed.user_source_ids_, row.source_user_id);
    interaction.movie_id = local_id(parsed.movie_source_ids_, row.source_movie_id);
    interaction.rating = row.rating;
    interaction.timestamp = row.timestamp;
    parsed.interactions_.push_back(interaction);
  }
  return parsed;
}

}  // namespace sagerec
