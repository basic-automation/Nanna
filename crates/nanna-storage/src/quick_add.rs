//! Quick-add: the board's only free-text entry (P25 decision 1).
//!
//! One line of text becomes one card. Todoist-style tokens fill fields and
//! everything else is the title:
//!
//! - `#label` — a label (any number; repeats are folded case-insensitively)
//! - `p1`..`p4` — priority
//! - `@member` — the assignee, as a handle the caller resolves against the
//!   board's roster ([`QuickAdd::assignee`] is the raw handle, never an id)
//! - a date phrase — the **defer** date (`due_at`, decision 10): `today`,
//!   `tomorrow`, a weekday (`friday`, `fri` — the next one strictly after
//!   today), `next <weekday>` (that day of next week), `next week` (the next
//!   Monday), `in N days` / `in N weeks`, `<month> <day>` / `<day> <month>`
//!   (the next such date, today included), or an ISO `YYYY-MM-DD`
//! - `{date phrase}` — the **deadline** (`deadline_at`), Todoist's brace
//!   syntax, same phrases inside
//!
//! A single-valued field given twice takes the **last** one, and the earlier
//! occurrence stays in the title as the words it was — "Plan monday's review
//! tomorrow" defers to tomorrow and keeps "monday's" (not a date word) and
//! nothing is lost. A brace group whose contents are not a date phrase is an
//! error rather than title text: braces say "deadline" too plainly to guess.
//!
//! The filter language (`task_filter`) is the other Todoist dialect and reads
//! `@` as a *label*; quick-add follows P25 decision 1, where `@` is a member.
//! Dates are computed against the caller's `today` — the store's days are UTC.

use chrono::{Datelike, Duration, NaiveDate, Weekday};

/// Maximum quick-add input length in bytes.
///
/// Bound justification: the store caps a title at 500 bytes and a card's
/// labels at 32 × 64 bytes; a line carrying both at their limits plus a few
/// date and member tokens fits in 4 KiB, so nothing the store would admit is
/// refused here, and the scan stays linear in a small input.
pub const QUICK_ADD_INPUT_MAX_BYTES: usize = 4096;

/// The furthest a relative date phrase (`in N days|weeks`) may reach.
///
/// Bound justification: ten years of days. A defer date further out than that
/// is a typo, not a plan, and the bound keeps `N` far inside `chrono`'s
/// representable range so date arithmetic can never overflow.
pub const QUICK_ADD_OFFSET_DAYS_MAX: i64 = 3660;

/// What one quick-add line asks for. Fields the line did not mention are
/// `None` / empty, so the router can complete them (decision 5).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct QuickAdd {
    /// Every word that was not a token, joined by single spaces.
    pub title: String,
    pub labels: Vec<String>,
    /// `1` (highest) ..= `4`, as the store and the filter's `p1`..`p4` read it.
    pub priority: Option<i64>,
    /// The `@` handle, without the `@`. Resolved by the caller.
    pub assignee: Option<String>,
    /// Defer date, `YYYY-MM-DD`.
    pub due_at: Option<String>,
    /// Deadline, `YYYY-MM-DD`.
    pub deadline_at: Option<String>,
}

/// Why a quick-add line was refused. Each message is shown to the human as is.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QuickAddError {
    #[error("quick-add exceeds {QUICK_ADD_INPUT_MAX_BYTES} bytes (got {0})")]
    TooLong(usize),
    #[error("a card needs a title — every word was a token")]
    NoTitle,
    #[error(
        "'{{{0}}}' is not a deadline — use a date phrase such as {{friday}}, {{in 3 days}} or {{2026-12-31}}"
    )]
    InvalidDeadline(String),
    #[error("'{0}' must land 0..={QUICK_ADD_OFFSET_DAYS_MAX} days from today")]
    OffsetTooFar(String),
}

/// One scanned piece of the line, with the words it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Text(String),
    Label(String),
    Priority(i64, String),
    Member(String, String),
    Date(NaiveDate, String),
    Deadline(NaiveDate, String),
}

impl Piece {
    /// The words this piece was typed as.
    fn source(&self) -> &str {
        match self {
            Self::Text(text) | Self::Label(text) => text,
            Self::Priority(_, source)
            | Self::Member(_, source)
            | Self::Date(_, source)
            | Self::Deadline(_, source) => source,
        }
    }
}

/// Parse one quick-add line against `today`.
///
/// # Errors
/// [`QuickAddError::TooLong`] past [`QUICK_ADD_INPUT_MAX_BYTES`],
/// [`QuickAddError::NoTitle`] when nothing is left for the title,
/// [`QuickAddError::InvalidDeadline`] for a brace group that is not a date
/// phrase, and [`QuickAddError::OffsetTooFar`] for `in N days|weeks` past
/// [`QUICK_ADD_OFFSET_DAYS_MAX`].
pub fn parse(text: &str, today: NaiveDate) -> Result<QuickAdd, QuickAddError> {
    if text.len() > QUICK_ADD_INPUT_MAX_BYTES {
        return Err(QuickAddError::TooLong(text.len()));
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    let pieces = scan(&words, today)?;
    debug_assert!(
        pieces.len() <= words.len(),
        "every piece consumes at least one word"
    );
    let quick_add = settle(&pieces);
    if quick_add.title.is_empty() {
        return Err(QuickAddError::NoTitle);
    }
    debug_assert!(
        !quick_add.title.contains("  "),
        "title words are single-spaced"
    );
    Ok(quick_add)
}

/// Turn the words into pieces, left to right. Every word lands in exactly one
/// piece, so `settle` can give any piece back to the title unchanged.
fn scan(words: &[&str], today: NaiveDate) -> Result<Vec<Piece>, QuickAddError> {
    let mut pieces = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        let (piece, used) = scan_one(&words[index..], today)?;
        debug_assert!(
            used >= 1 && index + used <= words.len(),
            "a piece takes 1..=rest words"
        );
        pieces.push(piece);
        index += used;
    }
    Ok(pieces)
}

/// The piece starting at `words[0]` and how many words it took.
fn scan_one(words: &[&str], today: NaiveDate) -> Result<(Piece, usize), QuickAddError> {
    let word = words[0];
    if word.starts_with('{')
        && let Some(found) = scan_deadline(words, today)?
    {
        return Ok(found);
    }
    if let Some(label) = word.strip_prefix('#').filter(|l| !l.is_empty()) {
        return Ok((Piece::Label(label.to_string()), 1));
    }
    if let Some(handle) = word.strip_prefix('@').filter(|h| !h.is_empty()) {
        return Ok((Piece::Member(handle.to_string(), word.to_string()), 1));
    }
    if let Some(priority) = priority_of(word) {
        return Ok((Piece::Priority(priority, word.to_string()), 1));
    }
    if let Some((date, used)) = date_phrase(words, today)? {
        return Ok((Piece::Date(date, words[..used].join(" ")), used));
    }
    Ok((Piece::Text(word.to_string()), 1))
}

/// A `{…}` group starting at `words[0]`. `None` when no word closes it, so
/// an unmatched `{` is ordinary title text.
fn scan_deadline(
    words: &[&str],
    today: NaiveDate,
) -> Result<Option<(Piece, usize)>, QuickAddError> {
    let Some(last) = words.iter().position(|w| w.ends_with('}')) else {
        return Ok(None);
    };
    let used = last + 1;
    let source = words[..used].join(" ");
    let inner = source
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or_default()
        .trim();
    let inner_words: Vec<&str> = inner.split_whitespace().collect();
    match date_phrase(&inner_words, today)? {
        Some((date, phrase_words)) if phrase_words == inner_words.len() => {
            Ok(Some((Piece::Deadline(date, source), used)))
        }
        _ => Err(QuickAddError::InvalidDeadline(inner.to_string())),
    }
}

/// `p1`..`p4`, case-insensitive.
fn priority_of(word: &str) -> Option<i64> {
    match word.to_ascii_lowercase().as_str() {
        "p1" => Some(1),
        "p2" => Some(2),
        "p3" => Some(3),
        "p4" => Some(4),
        _ => None,
    }
}

/// The date phrase starting at `words[0]`, if any, and how many words it is.
fn date_phrase(
    words: &[&str],
    today: NaiveDate,
) -> Result<Option<(NaiveDate, usize)>, QuickAddError> {
    let Some(first) = words.first().map(|w| w.to_ascii_lowercase()) else {
        return Ok(None);
    };
    let second = words.get(1).map(|w| w.to_ascii_lowercase());
    let found = match (first.as_str(), second.as_deref()) {
        ("today", _) => Some((today, 1)),
        ("tomorrow", _) => today.succ_opt().map(|d| (d, 1)),
        ("next", Some("week")) => Some((next_weekday(today, Weekday::Mon), 2)),
        ("next", Some(day)) => weekday_of(day).map(|day| (in_next_week(today, day), 2)),
        ("in", Some(_)) => in_offset(words, today)?,
        (word, Some(other)) if month_day(word, other, today).is_some() => {
            month_day(word, other, today).map(|d| (d, 2))
        }
        (word, _) => weekday_of(word)
            .map(|day| (next_weekday(today, day), 1))
            .or_else(|| {
                NaiveDate::parse_from_str(word, "%Y-%m-%d")
                    .ok()
                    .filter(|_| word.len() == 10)
                    .map(|d| (d, 1))
            }),
    };
    Ok(found)
}

/// `in N day(s)|week(s)`; `None` when the words are not that shape.
fn in_offset(
    words: &[&str],
    today: NaiveDate,
) -> Result<Option<(NaiveDate, usize)>, QuickAddError> {
    let (Some(count), Some(unit)) = (words.get(1), words.get(2)) else {
        return Ok(None);
    };
    let Ok(count) = count.parse::<i64>() else {
        return Ok(None);
    };
    let per_unit = match unit.to_ascii_lowercase().as_str() {
        "day" | "days" => 1,
        "week" | "weeks" => 7,
        _ => return Ok(None),
    };
    let days = count.saturating_mul(per_unit);
    if !(0..=QUICK_ADD_OFFSET_DAYS_MAX).contains(&days) {
        return Err(QuickAddError::OffsetTooFar(words[..3].join(" ")));
    }
    debug_assert!(
        days <= QUICK_ADD_OFFSET_DAYS_MAX,
        "bounded before the arithmetic"
    );
    Ok(today
        .checked_add_signed(Duration::days(days))
        .map(|d| (d, 3)))
}

/// `day` in the week after this one (weeks start on Monday): said on a
/// Wednesday, `next friday` is nine days out and `next monday` five.
fn in_next_week(today: NaiveDate, day: Weekday) -> NaiveDate {
    let monday = next_weekday(today, Weekday::Mon);
    let date = monday + Duration::days(i64::from(day.num_days_from_monday()));
    debug_assert!(
        date > today && date - today <= Duration::days(13),
        "within next week"
    );
    date
}

/// Years `month_day` looks ahead for a date that exists: the longest gap
/// between two Feb 29ths is 8 years (1896 → 1904).
const LEAP_DAY_YEARS_MAX: i32 = 8;

/// `march 30` or `30 march` (month full or three letters, day 1..=31, an
/// ordinal suffix allowed): the next such date on or after `today` — this
/// year's if it has not passed, else next year's. `None` for anything else,
/// a date no year has (`feb 30`) included, so those words stay in the title.
fn month_day(first: &str, second: &str, today: NaiveDate) -> Option<NaiveDate> {
    let (month, day) = month_of(first)
        .zip(day_of_month(second))
        .or_else(|| month_of(second).zip(day_of_month(first)))?;
    // Bounded: Feb 29 recurs within 8 years (leap years skip at most one
    // century year), and every other valid month/day within 2.
    let date = (0..=LEAP_DAY_YEARS_MAX)
        .filter_map(|ahead| NaiveDate::from_ymd_opt(today.year() + ahead, month, day))
        .find(|date| *date >= today);
    debug_assert!(date.is_none_or(|d| d >= today), "never in the past");
    date
}

/// A month name, full or three letters (`sept` too), as 1..=12.
fn month_of(word: &str) -> Option<u32> {
    const MONTHS: [&str; 12] = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let word = word.to_ascii_lowercase();
    if word.len() < 3 {
        return None;
    }
    let index = MONTHS
        .iter()
        .position(|m| *m == word || (word.len() <= 4 && m.starts_with(word.as_str())))?;
    u32::try_from(index + 1).ok()
}

/// `1`..=`31`, optionally with `st`/`nd`/`rd`/`th`.
fn day_of_month(word: &str) -> Option<u32> {
    let digits = word.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let suffix = &word[digits.len()..];
    if !matches!(
        suffix.to_ascii_lowercase().as_str(),
        "" | "st" | "nd" | "rd" | "th"
    ) {
        return None;
    }
    digits.parse::<u32>().ok().filter(|d| (1..=31).contains(d))
}

/// A weekday name, full or three letters.
fn weekday_of(word: &str) -> Option<Weekday> {
    match word {
        "monday" | "mon" => Some(Weekday::Mon),
        "tuesday" | "tue" => Some(Weekday::Tue),
        "wednesday" | "wed" => Some(Weekday::Wed),
        "thursday" | "thu" => Some(Weekday::Thu),
        "friday" | "fri" => Some(Weekday::Fri),
        "saturday" | "sat" => Some(Weekday::Sat),
        "sunday" | "sun" => Some(Weekday::Sun),
        _ => None,
    }
}

/// The next `day` strictly after `today` — saying "friday" on a Friday means
/// a week out, since a defer date of today is just "today".
fn next_weekday(today: NaiveDate, day: Weekday) -> NaiveDate {
    let ahead = (7 + i64::from(day.num_days_from_monday())
        - i64::from(today.weekday().num_days_from_monday()))
        % 7;
    let ahead = if ahead == 0 { 7 } else { ahead };
    debug_assert!((1..=7).contains(&ahead), "strictly after, at most a week");
    today + Duration::days(ahead)
}

/// Keep the last of each single-valued token; every other piece's words go
/// back into the title in their place.
fn settle(pieces: &[Piece]) -> QuickAdd {
    let last = |want: fn(&Piece) -> bool| pieces.iter().rposition(want);
    let priority_at = last(|p| matches!(p, Piece::Priority(..)));
    let member_at = last(|p| matches!(p, Piece::Member(..)));
    let date_at = last(|p| matches!(p, Piece::Date(..)));
    let deadline_at = last(|p| matches!(p, Piece::Deadline(..)));

    let mut quick_add = QuickAdd {
        title: String::new(),
        labels: Vec::new(),
        priority: None,
        assignee: None,
        due_at: None,
        deadline_at: None,
    };
    let mut title_words: Vec<&str> = Vec::new();
    for (index, piece) in pieces.iter().enumerate() {
        match piece {
            Piece::Label(label) => {
                if !quick_add
                    .labels
                    .iter()
                    .any(|l| l.eq_ignore_ascii_case(label))
                {
                    quick_add.labels.push(label.clone());
                }
            }
            Piece::Priority(p, _) if priority_at == Some(index) => quick_add.priority = Some(*p),
            Piece::Member(handle, _) if member_at == Some(index) => {
                quick_add.assignee = Some(handle.clone());
            }
            Piece::Date(date, _) if date_at == Some(index) => {
                quick_add.due_at = Some(date.format("%Y-%m-%d").to_string());
            }
            Piece::Deadline(date, _) if deadline_at == Some(index) => {
                quick_add.deadline_at = Some(date.format("%Y-%m-%d").to_string());
            }
            other => title_words.push(other.source()),
        }
    }
    quick_add.title = title_words.join(" ");
    quick_add
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Wednesday.
    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 10, 7).unwrap()
    }

    fn quick(text: &str) -> QuickAdd {
        parse(text, today()).unwrap()
    }

    #[test]
    fn plain_text_is_all_title() {
        let q = quick("  Write   the release notes ");
        assert_eq!(q.title, "Write the release notes");
        assert!(q.labels.is_empty() && q.priority.is_none() && q.assignee.is_none());
        assert!(q.due_at.is_none() && q.deadline_at.is_none());
    }

    #[test]
    fn every_token_kind_fills_its_field() {
        let q = quick("Fix the flaky test #ci #Rust p1 @Builder tomorrow {friday}");
        assert_eq!(q.title, "Fix the flaky test");
        assert_eq!(q.labels, vec!["ci", "Rust"]);
        assert_eq!(q.priority, Some(1));
        assert_eq!(q.assignee.as_deref(), Some("Builder"));
        assert_eq!(q.due_at.as_deref(), Some("2026-10-08"));
        assert_eq!(q.deadline_at.as_deref(), Some("2026-10-09"));
    }

    #[test]
    fn tokens_may_sit_anywhere_in_the_line() {
        let q = quick("p2 #home Clean @me the gutters");
        assert_eq!(q.title, "Clean the gutters");
        assert_eq!(q.priority, Some(2));
        assert_eq!(q.assignee.as_deref(), Some("me"));
    }

    #[test]
    fn repeated_labels_fold_case_insensitively_keeping_the_first_spelling() {
        let q = quick("Tidy #Home #home #HOME #garden");
        assert_eq!(q.labels, vec!["Home", "garden"]);
    }

    #[test]
    fn the_last_single_valued_token_wins_and_earlier_ones_stay_in_the_title() {
        let q = quick("Move p1 to p3 @ana or @ben today tomorrow");
        assert_eq!(q.priority, Some(3));
        assert_eq!(q.assignee.as_deref(), Some("ben"));
        assert_eq!(q.due_at.as_deref(), Some("2026-10-08"));
        assert_eq!(q.title, "Move p1 to @ana or today");
    }

    #[test]
    fn weekdays_mean_the_next_one_strictly_after_today() {
        assert_eq!(quick("a thursday").due_at.as_deref(), Some("2026-10-08"));
        assert_eq!(
            quick("a wed").due_at.as_deref(),
            Some("2026-10-14"),
            "today's weekday is a week out"
        );
        assert_eq!(quick("a Tuesday").due_at.as_deref(), Some("2026-10-13"));
        assert_eq!(quick("a next week").due_at.as_deref(), Some("2026-10-12"));
    }

    #[test]
    fn relative_and_iso_dates_parse() {
        assert_eq!(quick("a in 3 days").due_at.as_deref(), Some("2026-10-10"));
        assert_eq!(quick("a in 2 weeks").due_at.as_deref(), Some("2026-10-21"));
        assert_eq!(quick("a in 1 day").due_at.as_deref(), Some("2026-10-08"));
        assert_eq!(quick("a 2026-12-31").due_at.as_deref(), Some("2026-12-31"));
        assert_eq!(
            quick("a {in 10 days}").deadline_at.as_deref(),
            Some("2026-10-17")
        );
    }

    #[test]
    fn next_weekday_is_that_day_of_next_week() {
        assert_eq!(quick("a next friday").due_at.as_deref(), Some("2026-10-16"));
        assert_eq!(quick("a next monday").due_at.as_deref(), Some("2026-10-12"));
        assert_eq!(quick("a next wed").due_at.as_deref(), Some("2026-10-14"));
        assert_eq!(
            quick("a {next sunday}").deadline_at.as_deref(),
            Some("2026-10-18")
        );
        assert_eq!(quick("the next big thing").title, "the next big thing");
    }

    #[test]
    fn month_and_day_name_the_next_such_date() {
        assert_eq!(
            quick("a march 30").due_at.as_deref(),
            Some("2027-03-30"),
            "passed this year"
        );
        assert_eq!(quick("a 30 March").due_at.as_deref(), Some("2027-03-30"));
        assert_eq!(
            quick("a oct 7").due_at.as_deref(),
            Some("2026-10-07"),
            "today counts"
        );
        assert_eq!(quick("a Dec 25th").due_at.as_deref(), Some("2026-12-25"));
        assert_eq!(
            quick("a {sept 1}").deadline_at.as_deref(),
            Some("2027-09-01")
        );
        assert_eq!(
            quick("a feb 29").due_at.as_deref(),
            Some("2028-02-29"),
            "the next leap day"
        );
        assert_eq!(
            quick("May 4 be with you").due_at.as_deref(),
            Some("2027-05-04")
        );
        let q = quick("buy feb 30 tickets for 2 may");
        assert_eq!(q.title, "buy feb 30 tickets for");
        assert_eq!(
            q.due_at.as_deref(),
            Some("2027-05-02"),
            "an impossible date stays text"
        );
        assert_eq!(quick("ma 3 things").title, "ma 3 things");
    }

    #[test]
    fn near_misses_stay_in_the_title() {
        let q = quick("Log in to the box in 3 hours p5 # @ 2026-13-01 next time {unclosed");
        assert_eq!(
            q.title,
            "Log in to the box in 3 hours p5 # @ 2026-13-01 next time {unclosed"
        );
        assert!(q.priority.is_none() && q.due_at.is_none() && q.labels.is_empty());
        assert!(q.assignee.is_none() && q.deadline_at.is_none());
    }

    #[test]
    fn a_brace_group_that_is_not_a_date_is_refused() {
        assert_eq!(
            parse("Ship it {soonish}", today()),
            Err(QuickAddError::InvalidDeadline("soonish".to_string()))
        );
        assert_eq!(
            parse("Ship it {tomorrow please}", today()),
            Err(QuickAddError::InvalidDeadline(
                "tomorrow please".to_string()
            )),
            "the whole group must be the phrase"
        );
    }

    #[test]
    fn a_line_of_only_tokens_has_no_title() {
        assert_eq!(
            parse("#a p1 @me today", today()),
            Err(QuickAddError::NoTitle)
        );
        assert_eq!(parse("   ", today()), Err(QuickAddError::NoTitle));
    }

    #[test]
    fn offsets_and_input_are_bounded() {
        assert!(matches!(
            parse("a in 3661 days", today()),
            Err(QuickAddError::OffsetTooFar(_))
        ));
        assert!(matches!(
            parse("a in 523 weeks", today()),
            Err(QuickAddError::OffsetTooFar(_))
        ));
        assert_eq!(
            quick("a in 3660 days").due_at.as_deref(),
            Some("2036-10-14"),
            "3 leap days"
        );
        let long = "x".repeat(QUICK_ADD_INPUT_MAX_BYTES + 1);
        assert_eq!(
            parse(&long, today()),
            Err(QuickAddError::TooLong(QUICK_ADD_INPUT_MAX_BYTES + 1))
        );
    }

    #[test]
    fn negative_offsets_are_refused_rather_than_dated_in_the_past() {
        assert!(matches!(
            parse("a in -2 days", today()),
            Err(QuickAddError::OffsetTooFar(_))
        ));
    }
}
