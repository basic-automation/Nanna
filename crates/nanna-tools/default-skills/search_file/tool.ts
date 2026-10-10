export default {
  name: "search_file",
  version: "0.2.1",
  output: "memory",
  description: "Search within a single file for a pattern and return matching lines with surrounding context. Useful for finding specific functions, variables, or text in large files without reading the entire file. Returns line numbers so you can follow up with read_file for a broader view.",
  parameters: {
    type: "object",
    properties: {
      file_path: { type: "string", description: "Path to the file to search. Relative paths are resolved against the workspace directory." },
      pattern: { type: "string", description: "Search pattern (regex supported, case-insensitive by default)" },
      context_lines: { type: "integer", description: "Number of lines to show before and after each match. Default: 3" },
      max_results: { type: "integer", description: "Maximum number of matches to return. Default: 20" },
      case_sensitive: { type: "boolean", description: "Whether the search should be case-sensitive. Default: false" }
    },
    required: ["file_path", "pattern"]
  },
  execute: function(input) {
    var filePath = input.file_path || input.filePath || input.path || input.file || input.filename;
    if (!filePath) {
      // Structured, not thrown: a thrown script error reaches the model under
      // an "Execution failed:" prefix, which reads as breakage rather than as
      // a fixable call.
      return {
        content: "search_file: missing required parameter 'file_path'. Nothing was searched. " +
          "Call again with file_path set to the file you want to search.",
        success: false
      };
    }
    filePath = String(filePath);

    var searchPattern = input.pattern || input.query || input.search || input.keyword;
    if (!searchPattern) {
      return {
        content: "search_file: missing required parameter 'pattern'. Nothing was searched. " +
          "Call again with pattern set to the regex or literal text you are looking for.",
        success: false
      };
    }
    searchPattern = String(searchPattern);

    var ctx = parseInt(input.context_lines, 10);
    if (!isFinite(ctx) || ctx < 0) {
      ctx = 3;
    }
    var maxResults = parseInt(input.max_results, 10);
    if (!isFinite(maxResults) || maxResults < 1) {
      maxResults = 20;
    }
    var flags = input.case_sensitive ? "" : "i";

    var regex;
    var prefilter;
    try {
      regex = new RegExp(searchPattern, flags);
      // The prefilter runs against the whole file, the matcher against one
      // line. Without "m", `^` and `$` would anchor to the ends of the FILE, so
      // a pattern like "^fn main" would match line 40 in the per-line pass but
      // be rejected by the prefilter — a silent false negative. With "m" the
      // prefilter is a strict superset of "some line matches": it may let a
      // non-matching file through (costing one walk) but can never drop a
      // matching one.
      prefilter = new RegExp(searchPattern, flags + "m");
    } catch (e) {
      return {
        content: "search_file: '" + searchPattern + "' is not a valid regex (" + String(e) +
          "). Nothing was searched. Escape the regex metacharacters (. * + ? ( ) [ ] { } | \\) " +
          "or pass a plain literal.",
        success: false
      };
    }

    // ---------------------------------------------------------------------
    // Budgets. Each is derived from a measured constraint; none is a magic
    // cap. The measurements come from the ignored probes in
    // nanna-scripting/tests/search_file_skill.rs, taken through the real Boa
    // bridge on a debug build — the build the daemon actually runs.
    //
    // TIME is the primary bound, because time is the actual constraint: the
    // script deadline is 30s (ScriptedTool's default timeout_ms; this tool
    // declares no `timeout` override) and a script that overruns it returns
    // NOTHING — the failure is total, not degraded.
    //
    // The measurement that shapes everything below: `content.split("\n")` is
    // ~99% of the cost of a file-scanning skill, and its cost is the PRODUCT
    // of line count and string length, ~1e-8 x lines x chars seconds:
    //
    //   128 KiB /    653 lines   0.91s      128 KiB /  2,428 lines   3.41s
    //   128 KiB /  7,711 lines  10.29s      128 KiB / 26,215 lines  32.86s
    //   128 KiB / 65,536 lines  83.96s        1 MiB / 19,419 lines 184.06s
    //
    // Isolating the stages showed the split ITSELF is the whole cost — the
    // per-line loop over the result is free (32,768 lines: split alone 21.3s,
    // split+regex 23.2s, and an empty JS loop of the same count 70ms). So
    // neither a byte cap nor a line cap alone can bound it; the product must
    // be broken. Splitting in bounded slices does exactly that, turning
    // O(lines x length) into O(lines x chunk):
    //
    //   1 MiB / 19,419 lines:  one split 184.06s  ->  4 KiB slices 3.05s (60x)
    //   128 KiB / 65,536 lines: one split 82.53s  ->  4 KiB slices 2.68s (31x)
    //
    // The reserve is the OVERSHOOT, not a round fraction. With the walk sliced,
    // the largest step that cannot be interrupted is the single readFile
    // (~1.3s at the cap below); one slice is ~0.1s worst case. Reserve =
    // 30000 - 20000 = 10,000ms, which covers that read even on a machine
    // three times slower than the one measured (3 x 1.3s = 3.9s).
    var SCRIPT_DEADLINE_MS = 30000;
    var WORK_BUDGET_MS = 20000;

    // READ — pulling a file's text across the bridge measured 7.7 MB/s, and
    // the read cannot be interrupted once entered, so it has to fit the
    // reserve with room to spare: 10,000ms / 3 (slower machine) leaves ~3.3s,
    // and 1.3s of that at 7.7 MB/s is ~10 MB. (This is the cap the skill
    // already had as a bare constant; it survives the derivation unchanged.)
    var MAX_READ_BYTES = 10 * 1024 * 1024;

    // SLICE — the width of one split. Measured above: 4 KiB slices made a
    // 1 MiB file 60x cheaper to walk. It also bounds the one step the clock
    // cannot preempt: a slice holds at most SPLIT_SLICE_CHARS/2 lines (every
    // line "x\n"), so its split costs at most ~1e-8 x 2,048 x 4,096 ~ 0.08s.
    var SPLIT_SLICE_CHARS = 4096;

    // WALK — the file size worth doing line work on at all. A 1 MiB file of
    // ordinary source measured 3.05s to walk in slices, a sixth of the work
    // budget. Past that the honest answer is faster and more useful than a
    // partial one: the prefilter already proved whether the pattern is there,
    // so we report THAT and point at ripgrep for the line numbers.
    var MAX_WALK_CHARS = 1024 * 1024;

    // LINES — a backstop for the case where the clock is unusable, and the
    // bound on how much of the file is held as an array of JS strings. At the
    // slowest measured slice-walk rate (19,419 lines of a 1 MiB file in
    // 3.05s = ~6,400 lines/s), half the work budget is ~64,000 lines; round
    // down to 60,000.
    var MAX_WALK_LINES = 60000;

    // OUTPUT — this tool declares output:"memory", so its result is not cut at
    // the ~2KB context boundary the context-routed tools face. It is split
    // into MEMORY_CHUNK_MAX_CHARS (3200-char) chunks and EVERY chunk becomes
    // its own memory row: one embedding, one similarity search, one write. So
    // result size has a cost denominated in chunks, and the honest ceiling is
    // what this tool's own defaults were designed to cost. A default call is
    // max_results 20 matches at context_lines 3, i.e. 7 lines each; at the
    // repo's 100-column formatting convention a rendered line is 100 chars
    // plus the ~10-char " > 1234 | " gutter, so a section is ~770 chars, and
    // 20 of them plus a header is ~15,500 chars ~ 4.8 chunks. Chunking is by
    // whole chunk, so round up: 5.
    var MEMORY_CHUNK_CHARS = 3200;
    var MAX_OUTPUT_CHARS = 5 * MEMORY_CHUNK_CHARS;

    var startedAt = Date.now();
    // If the engine ever serves a non-numeric clock, the time budget silently
    // becomes a no-op — which is exactly the unbounded behaviour being fixed.
    // Detect it here so the other bounds carry the load and the note below
    // tells the truth about which bound applied.
    var clockOk = isFinite(startedAt);

    // A missing file is a structured answer, not a throw.
    var stat;
    try {
      stat = Nanna.stat(filePath);
    } catch (e) {
      return {
        content: "search_file: '" + filePath + "' does not exist (or is unreadable). Nothing was " +
          "searched. Check the path with exec `ls` first.",
        success: false
      };
    }
    if (stat.is_dir) {
      return {
        content: "search_file: '" + filePath + "' is a directory, not a file. Nothing was " +
          "searched. Use code_search to search across a directory tree.",
        success: false
      };
    }
    if (stat.size > MAX_READ_BYTES) {
      return {
        content: "search_file: '" + filePath + "' is " + formatSize(stat.size) + ", over the " +
          formatSize(MAX_READ_BYTES) + " read budget (pulling it across the bridge would spend " +
          "the reserve this skill keeps against the " + SCRIPT_DEADLINE_MS + "ms script " +
          "deadline). Nothing was searched. Use exec `rg -n \"" + searchPattern + "\" \"" +
          filePath + "\"` — ripgrep streams the file instead of loading it whole.",
        success: false
      };
    }

    // read_to_string rejects non-UTF-8 bytes, so a binary file usually fails
    // HERE rather than at the NUL check below. Both paths are answers.
    var content;
    try {
      content = Nanna.readFile(filePath);
    } catch (e) {
      return {
        content: "search_file: '" + filePath + "' could not be read as text (" + String(e) +
          "). Nothing was searched. It is most likely a binary or non-UTF-8 file; use exec " +
          "`rg -a` if you really need to search its bytes.",
        success: false
      };
    }
    if (content.substring(0, 512).indexOf("\0") >= 0) {
      return {
        content: "search_file: '" + filePath + "' looks binary (NUL bytes in the first 512 " +
          "chars). Nothing was searched. Use exec `rg -a` if you really need to search its bytes.",
        success: false
      };
    }

    // Whole-string prefilter. Testing the regex against the entire file is
    // effectively free — the same cost as reading it — while walking that text
    // line by line is everything. A file with no match must therefore never
    // reach the walk, and in real use most files do not match. This one line
    // is the difference between a search that finishes and one that eats the
    // deadline.
    if (!prefilter.test(content)) {
      return {
        content: "No matches for \"" + searchPattern + "\" in " + filePath + " — the whole file " +
          "(" + formatSize(content.length) + ") was tested as one string and does not contain " +
          "the pattern anywhere. The search SUCCEEDED and is complete. (Line numbers are not " +
          "reported for a miss: splitting the file into lines is the expensive step, and there " +
          "is nothing to point at.)",
        success: true
      };
    }

    // Past here the file DOES contain the pattern. Every bound below can cost
    // the LOCATION of the match, but none may cost that fact.
    if (content.length > MAX_WALK_CHARS) {
      return {
        content: filePath + " CONTAINS \"" + searchPattern + "\", but at " +
          formatSize(content.length) + " it is larger than the " + formatSize(MAX_WALK_CHARS) +
          " line-context budget, so no line numbers or surrounding lines were rendered " +
          "(walking a file that size line by line would spend most of the " +
          SCRIPT_DEADLINE_MS + "ms script deadline on this one call). The search SUCCEEDED — " +
          "the match is real, only its location is missing. Use exec `rg -n \"" + searchPattern +
          "\" \"" + filePath + "\"` for line numbers, or narrow the pattern.",
        success: true
      };
    }

    // ---------------------------------------------------------------------
    // Sliced walk. Produces exactly the array `content.split("\n")` would, but
    // in interruptible steps, and scans each slice as it lands so a match near
    // the top of a big file costs only the lines above it.
    var lines = [];
    var matchIndices = [];
    var scanned = 0;
    var pos = 0;
    var timeCapped = false;
    var lineCapped = false;

    while (pos < content.length) {
      if (clockOk && Date.now() - startedAt > WORK_BUDGET_MS) {
        timeCapped = true;
        break;
      }
      if (lines.length >= MAX_WALK_LINES) {
        lineCapped = true;
        break;
      }

      // Extend the slice to the end of the line it lands in, so a line is
      // never cut in half across two splits.
      var end = pos + SPLIT_SLICE_CHARS;
      if (end < content.length) {
        var nl = content.indexOf("\n", end);
        end = nl < 0 ? content.length : nl + 1;
      } else {
        end = content.length;
      }
      var part = content.slice(pos, end);
      pos = end;

      var got = part.split("\n");
      // A slice that ended on a newline leaves a trailing "" that is not a
      // line — it is the start of the next slice's first line. At EOF it IS a
      // line (the empty segment after a trailing newline), matching split.
      if (pos < content.length && got.length > 0 && got[got.length - 1] === "") {
        got.pop();
      }
      // A CRLF file's lines carry their `\r` after a split on `\n`, and
      // without the `m` flag `$` does not match before it: `\{$` found nothing
      // in a CRLF file (and the miss was blamed on a pattern spanning lines).
      for (var g = 0; g < got.length; g++) {
        var line = got[g];
        if (line.length > 0 && line.charCodeAt(line.length - 1) === 13) line = line.substring(0, line.length - 1);
        lines.push(line);
      }

      // Scan what just landed. The per-line loop is free next to the split.
      while (scanned < lines.length && matchIndices.length < maxResults) {
        if (regex.test(lines[scanned])) {
          matchIndices.push(scanned);
        }
        scanned++;
      }

      // Once we have all the matches we were asked for AND enough trailing
      // lines to render their context, the rest of the file cannot change the
      // answer — stop reading it.
      if (matchIndices.length >= maxResults &&
          lines.length > matchIndices[matchIndices.length - 1] + ctx) {
        break;
      }
    }
    // `"".split("\n")` is [""], but the loop above never runs for an empty
    // file. Keep the two equivalent.
    if (content.length === 0) {
      lines.push("");
    }
    // Scan anything the loop appended but did not reach — only the empty-file
    // line above, since the loop scans each slice as it lands. A no-op
    // otherwise, and it keeps "what was walked" and "what was scanned" equal
    // on every path.
    while (scanned < lines.length && matchIndices.length < maxResults) {
      if (regex.test(lines[scanned])) {
        matchIndices.push(scanned);
      }
      scanned++;
    }

    var walkedLines = lines.length;
    var complete = pos >= content.length;

    // Every bound that shaped the answer says so, in the answer.
    var notes = [];
    if (timeCapped) {
      notes.push("STOPPED: spent the " + WORK_BUDGET_MS + "ms work budget (kept under the " +
        SCRIPT_DEADLINE_MS + "ms script deadline so the result comes back instead of the whole " +
        "call timing out) after " + walkedLines + " line(s), " + formatSize(pos) + " of " +
        formatSize(content.length) + ". The search SUCCEEDED over what it reached and the rest " +
        "of the file was not examined — narrow the pattern, or use exec `rg -n \"" +
        searchPattern + "\" \"" + filePath + "\"`.");
    }
    if (lineCapped) {
      notes.push("STOPPED: hit the " + MAX_WALK_LINES + "-line walk budget after " +
        formatSize(pos) + " of " + formatSize(content.length) + " (holding more of the file as " +
        "individual lines costs more than the whole call is worth). The search SUCCEEDED over " +
        "what it reached and the rest of the file was not examined — use exec `rg -n \"" +
        searchPattern + "\" \"" + filePath + "\"` for the whole file.");
    }

    if (matchIndices.length === 0) {
      // The prefilter proved the pattern is in this file, so an empty per-line
      // result means either a bound cut the walk short or the match spans a
      // line break. Reporting "no matches" here would be a lie.
      var head;
      if (!complete) {
        head = filePath + " CONTAINS \"" + searchPattern + "\", but no matching line was " +
          "reached before the walk was stopped — see below.";
      } else {
        head = filePath + " CONTAINS \"" + searchPattern + "\" as whole text, but no SINGLE " +
          "line matches it (" + walkedLines + " lines checked). The pattern most likely spans a " +
          "line break, which this line-by-line search cannot report. The search SUCCEEDED — use " +
          "read_file to look at the file directly.";
      }
      return {
        content: notes.length > 0 ? head + "\n\n" + notes.join("\n") : head,
        success: true
      };
    }

    // Build output with context, merging overlapping regions. Each line index
    // is emitted at most once (regions merge on lastShownEnd), so the render
    // costs at most one pass over the lines already walked.
    var padLen = String(walkedLines).length;
    var sections = [];
    var lastShownEnd = -1;
    var renderedChars = 0;
    var outputCapped = false;

    for (var mi = 0; mi < matchIndices.length; mi++) {
      var matchIdx = matchIndices[mi];
      var start = Math.max(0, matchIdx - ctx);
      var stop = Math.min(walkedLines - 1, matchIdx + ctx);
      var merging = start <= lastShownEnd + 1 && sections.length > 0;
      if (merging) {
        start = lastShownEnd + 1;
        if (start > stop) continue; // already shown
      }

      var rendered = [];
      for (var i = start; i <= stop; i++) {
        var marker = i === matchIdx ? " >" : "  ";
        var lineNum = String(i + 1);
        while (lineNum.length < padLen) lineNum = " " + lineNum;
        rendered.push(marker + " " + lineNum + " | " + lines[i]);
      }
      var cost = joinedLength(rendered);
      // The first section always renders: a result with a budget note and no
      // match at all would be strictly less useful than one oversized section.
      if (renderedChars + cost > MAX_OUTPUT_CHARS && sections.length > 0) {
        outputCapped = true;
        break;
      }
      renderedChars += cost;

      if (merging) {
        var section = sections[sections.length - 1];
        for (var r = 0; r < rendered.length; r++) {
          section.push(rendered[r]);
        }
      } else {
        if (sections.length > 0) {
          sections[sections.length - 1].push("  " + repeat("·", padLen + 3));
        }
        sections.push(rendered);
      }
      lastShownEnd = stop;
    }

    if (outputCapped) {
      notes.push("STOPPED: the next match section would pass the " + MAX_OUTPUT_CHARS +
        "-char result budget (" + (MAX_OUTPUT_CHARS / MEMORY_CHUNK_CHARS) + " memory chunks, " +
        "what a default call to this tool costs to store). Every section below is complete; " +
        "further matches were not appended. The search SUCCEEDED — lower context_lines or " +
        "max_results, or narrow the pattern.");
    }
    if (matchIndices.length >= maxResults) {
      notes.push("Stopped at the requested max_results of " + maxResults +
        " matches; more may exist.");
    }
    if (!clockOk) {
      notes.push("Note: this engine served no usable clock, so the time budget was not " +
        "enforced; the read, walk-size, line and result budgets still were.");
    }

    var shown = 0;
    for (var mj = 0; mj < matchIndices.length; mj++) {
      if (matchIndices[mj] <= lastShownEnd) {
        shown++;
      }
    }
    var partial = timeCapped || lineCapped || outputCapped;
    var scope = complete
      ? walkedLines + " lines"
      : "first " + walkedLines + " lines of " + formatSize(content.length);
    var output = "Found " + shown + " match" + (shown > 1 ? "es" : "") + " in " + filePath +
      " (" + scope + ")" + (partial ? " (PARTIAL — see notes at the end)" : "") + "\n\n";
    for (var si = 0; si < sections.length; si++) {
      output += sections[si].join("\n") + "\n";
    }
    if (notes.length > 0) {
      output += "\n" + notes.join("\n");
    }

    return { content: output, success: true };
  }
}

// Cost of a group of rendered lines once joined with "\n", counted before the
// group is appended so the budget is checked against what it would actually add.
function joinedLength(rendered) {
  var n = 0;
  for (var i = 0; i < rendered.length; i++) {
    n += rendered[i].length + 1;
  }
  return n;
}

function repeat(ch, n) {
  var s = "";
  for (var i = 0; i < n; i++) s += ch;
  return s;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + "B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + "KB";
  return (bytes / (1024 * 1024)).toFixed(1) + "MB";
}
