export default {
  name: "code_search",
  version: "0.3.2",
  output: "memory",
  description: "Search for a pattern across files in a directory tree. Returns matching lines with context. Supports regex patterns, a filename glob filter, and a depth bound.",
  parameters: {
    type: "object",
    properties: {
      pattern: { type: "string", description: "Search pattern (regex supported, case-insensitive)" },
      path: { type: "string", description: "Directory to search in. Default: current directory" },
      file_pattern: { type: "string", description: "Glob-style filter for filenames (e.g. '*.rs', '*.ts'). Narrows the scan onto the files you care about, so the budget goes further." },
      context_lines: { type: "integer", description: "Number of context lines before/after match. Default: 2" },
      max_results: { type: "integer", description: "Maximum number of matches. Default: 50" },
      depth: { type: "integer", description: "How many directory levels below `path` to descend. Default: 10" }
    },
    required: ["pattern"]
  },
  execute: function(input) {
    var searchPattern = input.pattern || input.query || input.search || input.regex;
    if (!searchPattern) {
      // Structured, not thrown: a thrown script error reaches the model under
      // an "Execution failed:" prefix, which reads as breakage rather than as
      // a fixable call.
      return {
        content: "code_search: missing required parameter 'pattern'. Nothing was searched. " +
          "Call again with pattern set to the regex or literal text you are looking for.",
        success: false
      };
    }
    searchPattern = String(searchPattern);

    var searchPath = String(input.path || input.dir || input.directory || ".");
    // Trim trailing separators so joined child paths never double up.
    while (searchPath.length > 1 && (searchPath.charAt(searchPath.length - 1) === "/" ||
           searchPath.charAt(searchPath.length - 1) === "\\")) {
      searchPath = searchPath.slice(0, searchPath.length - 1);
    }

    var ctx = parseInt(input.context_lines, 10);
    if (!isFinite(ctx) || ctx < 0) {
      ctx = 2;
    }
    var maxResults = parseInt(input.max_results, 10);
    if (!isFinite(maxResults) || maxResults < 1) {
      maxResults = 50;
    }
    var maxDepth = parseInt(input.depth, 10);
    if (!isFinite(maxDepth) || maxDepth < 1) {
      // Matches the bridge's own recursive-walk ceiling, so replacing the
      // recursive listDir with an explicit level-by-level walk does not
      // silently change which files are reachable.
      maxDepth = 10;
    }
    var filePattern = input.file_pattern || input.filePattern || null;

    var regex;
    var prefilter;
    try {
      regex = new RegExp(searchPattern, "i");
      // The prefilter runs against a whole file, the matcher against one line.
      // Without "m", `^` and `$` would anchor to the ends of the FILE, so a
      // pattern like "^fn main" would match line 40 in the per-line pass but
      // be rejected by the prefilter — a silent false negative. With "m" the
      // prefilter is a strict superset of "some line matches": it may let a
      // non-matching file through (costing one line walk) but can never drop
      // a matching one.
      prefilter = new RegExp(searchPattern, "im");
    } catch (e) {
      return {
        content: "code_search: '" + searchPattern + "' is not a valid regex (" + String(e) +
          "). Nothing was searched. Escape the regex metacharacters (. * + ? ( ) [ ] { } | \\) " +
          "or pass a plain literal.",
        success: false
      };
    }

    // ---------------------------------------------------------------------
    // Budgets. Each is derived from a measured constraint; none is a magic
    // cap. The measurements come from the ignored throughput test in
    // nanna-scripting/tests/code_search_skill.rs, taken through the real Boa
    // bridge on a debug build — the build the daemon actually runs.
    //
    // TIME is the primary bound, because time is the actual constraint: the
    // script deadline is 30s (ScriptedTool's default timeout_ms; this tool
    // declares no `timeout` override) and a script that overruns it returns
    // NOTHING — the failure is total, not degraded. Bounding wall-clock
    // directly, instead of proxying it through a byte count, is also the only
    // bound that holds across machines and file shapes: measured cost per
    // file varies by ~75x with shape alone (11.6 MB read in 1.5s; the same
    // 11.6 MB split into lines, 112s).
    //
    // The reserve is the OVERSHOOT, not a round fraction — the cost of the
    // longest stretch that runs AFTER a budget check and cannot itself be
    // interrupted. With the per-file walk sliced (SPLIT_SLICE_CHARS below)
    // that stretch is no longer the render: it is one directory listing
    // bounded by MAX_ENTRIES (12,000 entries at the measured 8,800/s ≈ 1.4s),
    // then, for the first file under it, a read bounded by MAX_FILE_BYTES
    // (~0.13s at the measured 7.7 MB/s) plus a whole-string prefilter over
    // the same bytes (~0.11s) plus the one slice in flight when the clock
    // trips (~0.08s, derived below) ≈ 1.7s. Triple that so a machine three
    // times slower than the one measured still lands inside the deadline
    // (30000 - 3 x 1700 = 24,900); 20,000 keeps the slack the earlier,
    // un-sliced reserve was sized for and costs nothing to hold.
    var SCRIPT_DEADLINE_MS = 30000;
    var WORK_BUDGET_MS = 20000;

    // ENTRIES — every directory entry crosses the bridge as its own JS
    // object, and marshalling a whole workspace's worth (hundreds of
    // thousands under node_modules/.git/target) is what blew the deadline
    // before a single line of script ran. Measured at 8,800 entries/s;
    // discovery only feeds the scan, so it gets a tenth of the work budget:
    // 1.5s x 8,800 = 13,200, rounded down.
    var MAX_ENTRIES = 12000;

    // READ — pulling a file's text across the bridge measured 7.7 MB/s. No
    // single read may cost more than ~1% of the work budget (0.15s), so the
    // largest file worth reading whole is ~1.15 MB, rounded down to 1 MiB.
    var MAX_FILE_BYTES = 1024 * 1024;

    // SLICE — the width of one split. Splitting a file into lines is the
    // expensive step, and the measurement that pins it down is that
    // `content.split("\n")` costs the PRODUCT of line count and string
    // length, ~1e-8 x lines x chars seconds. At a FIXED 128 KiB:
    //
    //     653 lines  0.91s     2,428 lines  3.41s     7,711 lines 10.29s
    //  26,215 lines 32.86s    65,536 lines 83.96s
    //
    // So a byte cap cannot bound this: 128 KiB of one-char lines (minified
    // JS, a CSV, a log, a wrapped base64 blob) passes any byte cap and then
    // spends ~84s inside ONE uninterruptible split — past the 30s deadline,
    // which means the whole call returns nothing. Isolating the stages showed
    // the split ITSELF is the entire cost, so no cheaper per-line loop helps
    // (32,768 one-char lines: split alone 21.3s, split+regex 23.2s, an empty
    // JS loop of the same count 70ms).
    //
    // Splitting in bounded slices breaks the product, turning
    // O(lines x length) into O(lines x slice):
    //
    //   1 MiB / 19,419 lines:  one split 184.06s  ->  4 KiB slices 3.05s (60x)
    //   128 KiB / 65,536 lines: one split 82.53s  ->  4 KiB slices 2.68s (31x)
    //   128 KiB /  2,428 lines: one split  2.26s  ->  4 KiB slices 0.13s (17x)
    //
    // 4 KiB is also what makes the walk preemptible at all: a slice holds at
    // most SPLIT_SLICE_CHARS/2 lines (every line "x\n"), so one slice costs at
    // most ~1e-8 x 2,048 x 4,096 ≈ 0.08s, and the work budget can be checked
    // between slices instead of only between files.
    var SPLIT_SLICE_CHARS = 4096;

    // RENDER — the file size worth doing line work on at all. Once the walk
    // is sliced its cost is ~1e-8 x lines x SPLIT_SLICE_CHARS, and BYTES NOW
    // BOUND LINES: the densest a file can be packed is one line per two chars
    // ("x\n"), so 128 KiB holds at most 65,536 lines and its walk costs at
    // most ~1e-8 x 65,536 x 4,096 ≈ 2.7s — an eighth of the work budget, the
    // most one file may take. That is the same 128 KiB and the same "eighth
    // of the budget" this cap already carried, but it is only TRUE with the
    // slicing above: measured against one split, that same worst-case file
    // costs 84s. Larger files are still searched (the whole-string prefilter
    // below is cheap); only their line context is skipped, and the skip
    // announces itself.
    var RENDER_MAX_CHARS = 128 * 1024;

    // OUTPUT — this tool declares output:"memory", so its result is not cut
    // at the ~2KB context boundary the context-routed tools face. It is split
    // into MEMORY_CHUNK_MAX_CHARS (3200-char) chunks and EVERY chunk becomes
    // its own memory row: one embedding, one similarity search, one write. So
    // result size has a cost denominated in chunks, and the honest ceiling is
    // what this tool's own defaults were designed to cost. A default call is
    // max_results 50 sections at context_lines 2, i.e. 5 lines each; at the
    // repo's 100-column formatting convention a rendered line is 100 chars
    // plus the 7-char "> 123: " gutter, so a section is ~535 chars plus a
    // ~45-char file header ≈ 580, and 50 of them ≈ 29,000 chars ≈ 9.1 chunks.
    // Chunking is by whole chunk, so round up: 10.
    var MEMORY_CHUNK_CHARS = 3200;
    var MAX_OUTPUT_CHARS = 10 * MEMORY_CHUNK_CHARS;

    // Heavy/noise directories are never descended into. Mirrors the bridge's
    // recursive-walk ignore list — which a non-recursive, level-by-level walk
    // no longer gets for free — and the explore skill's. Every skip is
    // announced in the output.
    var SKIP_DIRS = [
      "node_modules", ".git", "target", "dist", "build", ".venv", "venv",
      "__pycache__", ".next", ".nuxt", ".cache"
    ];

    // Extensions whose bytes cannot contain source text. Filtered BEFORE the
    // read, because the marshal is the expensive half: a 900KB .png costs the
    // same to pull across the bridge as 900KB of code, and then is discarded
    // by the NUL check one line later.
    var BINARY_EXTS = [
      "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "svgz", "tif", "tiff",
      "pdf", "zip", "gz", "tgz", "bz2", "xz", "7z", "rar", "jar", "class",
      "exe", "dll", "so", "dylib", "obj", "o", "a", "lib", "pdb", "bin",
      "mp3", "mp4", "wav", "ogg", "webm", "mov", "avi", "flac",
      "ttf", "otf", "woff", "woff2", "eot",
      "db", "sqlite", "sqlite3", "wasm", "pyc", "pyd"
    ];

    var startedAt = Date.now();
    // If the engine ever serves a non-numeric clock, the time budget silently
    // becomes a no-op — which is exactly the unbounded behaviour being fixed.
    // Detect it here so the other bounds carry the load and the note below
    // tells the truth about which bound applied.
    var clockOk = isFinite(startedAt);
    // Absolute cutoff, so the sliced walk below can be preempted without
    // having to carry `startedAt` and the budget separately. Null means the
    // clock is unusable and the walk falls back to the size bounds alone.
    var deadlineAt = clockOk ? startedAt + WORK_BUDGET_MS : null;

    var visited = 0;
    var entriesCapped = false;
    var timeCapped = false;
    var outputCapped = false;
    var bytesRead = 0;
    var filesScanned = 0;
    var candidates = 0;
    var matchedFiles = 0;
    var notRendered = 0;
    var skippedLarge = 0;
    var skippedBinary = 0;
    var readErrors = 0;
    var unreadableDirs = 0;
    var skippedCounts = {};
    var skippedAny = false;
    var totalMatches = 0;
    var results = [];

    // Level-by-level walk with bounded, NON-recursive listings, interleaved
    // with the scan. The FIFO queue finishes each depth before starting the
    // next, so whichever budget trips first, what drops is the deepest —
    // least likely to be the code the caller meant — rather than an arbitrary
    // slice of a depth-first walk.
    var stop = false;
    var queue = [{ path: searchPath, rel: "", depth: 1 }];
    while (queue.length > 0 && !stop) {
      if (clockOk && Date.now() - startedAt > WORK_BUDGET_MS) {
        timeCapped = true;
        break;
      }
      var cur = queue.shift();
      var entries;
      try {
        // Request one more than the remaining budget: an overflow-length
        // return proves the directory holds more than we will visit.
        entries = Nanna.listDir(cur.path, false, MAX_ENTRIES - visited + 1);
      } catch (e) {
        if (cur.depth === 1) {
          // A FILE passed as `path` is the commonest cause, and its os error
          // ("The directory name is invalid", ENOTDIR) teaches nothing — one
          // stat, on an already-failed call, turns it into the correction.
          // The stat has its own try so this error path can never throw.
          if (pathIsFile(searchPath)) {
            return {
              content: "code_search: \"" + searchPath + "\" exists but is a FILE, not a " +
                "directory — use search_file to search it (or read_file to see its contents). " +
                "Nothing was searched.",
              success: false
            };
          }
          return {
            content: "code_search failed: cannot list \"" + searchPath + "\": " + String(e) +
              ". Nothing was searched. Check that the path exists, is a directory, and is " +
              "readable.",
            success: false
          };
        }
        unreadableDirs++;
        continue;
      }

      for (var i = 0; i < entries.length; i++) {
        if (stop) {
          break;
        }
        if (visited >= MAX_ENTRIES) {
          entriesCapped = true;
          stop = true;
          break;
        }
        visited++;

        var e = entries[i];
        var name = String(e.name);
        var childPath = joinPath(cur.path, name);
        var childRel = cur.rel ? cur.rel + "/" + name : name;

        if (e.entry_type === "dir") {
          if (SKIP_DIRS.indexOf(name) >= 0) {
            skippedCounts[name] = (skippedCounts[name] || 0) + 1;
            skippedAny = true;
          } else if (cur.depth < maxDepth) {
            queue.push({ path: childPath, rel: childRel, depth: cur.depth + 1 });
          }
          continue;
        }
        if (e.entry_type !== "file") {
          continue; // symlinks are not followed: they re-enter skipped trees
        }
        if (filePattern && !matchGlob(childRel, filePattern)) {
          continue;
        }

        candidates++;
        var size = e.size || 0;
        if (size >= MAX_FILE_BYTES) {
          skippedLarge++;
          continue;
        }
        if (hasBinaryExt(name, BINARY_EXTS)) {
          skippedBinary++;
          continue;
        }
        if (clockOk && Date.now() - startedAt > WORK_BUDGET_MS) {
          timeCapped = true;
          stop = true;
          break;
        }

        var content;
        try {
          content = Nanna.readFile(childPath);
        } catch (err) {
          readErrors++;
          continue;
        }
        // `size` is authoritative when the listing carried metadata; when it
        // did not (0), the text we just pulled across IS the measurement.
        bytesRead += size > 0 ? size : content.length;
        filesScanned++;

        if (content.substring(0, 512).indexOf("\0") >= 0) {
          skippedBinary++;
          continue;
        }

        // Whole-string prefilter. Testing the regex against the entire file
        // is effectively free (11.6 MB measured at 1.3s, the same cost as
        // reading it), while walking that text line by line is everything
        // else — 112s for the same bytes as one split per file, and still the
        // dominant per-file cost once sliced. A file with no match must
        // therefore never reach the walk, and in a real search almost none
        // does. This one line is the difference between a search that
        // finishes and one that eats the deadline.
        if (!prefilter.test(content)) {
          continue;
        }
        matchedFiles++;
        if (content.length > RENDER_MAX_CHARS) {
          notRendered++;
          continue;
        }

        // Split and scan in bounded slices rather than one `split("\n")`.
        // Same lines, same matches, but the cost stops being the product of
        // the file's line count and its length, and the work budget can
        // preempt the walk part-way instead of only between files.
        var walk = walkLines(content, regex, maxResults - totalMatches, ctx,
          SPLIT_SLICE_CHARS, deadlineAt);
        var lines = walk.lines;
        var matches = walk.matches;
        totalMatches += matches.length;

        if (matches.length > 0) {
          var section = formatFileMatches(childPath, lines, matches, ctx);
          if (outputChars(results) + section.length > MAX_OUTPUT_CHARS) {
            outputCapped = true;
            stop = true;
            break;
          }
          results.push(section);
        }

        // Rendered first, then stop: a walk cut off by the clock still found
        // real matches in the lines it reached, and dropping them would make
        // the budget cost more than it saves.
        if (walk.timeCapped) {
          timeCapped = true;
          stop = true;
          break;
        }

        if (totalMatches >= maxResults) {
          stop = true;
          break;
        }
      }
    }

    // Every bound that shaped the answer says so, in the answer.
    var notes = [];
    if (skippedAny) {
      var skipParts = [];
      for (var s = 0; s < SKIP_DIRS.length; s++) {
        var sk = SKIP_DIRS[s];
        if (skippedCounts[sk]) {
          skipParts.push(skippedCounts[sk] > 1 ? sk + " x" + skippedCounts[sk] : sk);
        }
      }
      notes.push("Skipped noise dirs (never descended): " + skipParts.join(", "));
    }
    if (skippedLarge > 0) {
      notes.push("Skipped " + skippedLarge + " file(s) at or above " +
        formatSize(MAX_FILE_BYTES) + " — too large to pull across the bridge whole.");
    }
    if (skippedBinary > 0) {
      notes.push("Skipped " + skippedBinary + " binary/non-text file(s).");
    }
    if (readErrors > 0) {
      notes.push("Skipped " + readErrors + " file(s) that could not be read.");
    }
    if (unreadableDirs > 0) {
      notes.push("Skipped " + unreadableDirs + " subdirector" +
        (unreadableDirs === 1 ? "y" : "ies") + " that could not be listed.");
    }
    if (notRendered > 0) {
      notes.push(notRendered + " file(s) CONTAIN the pattern but are larger than " +
        formatSize(RENDER_MAX_CHARS) + ", so their line context was not rendered (walking a " +
        "file line by line costs its line count, and above that size one file could take more " +
        "than the eighth of the work budget any single file is allowed). They are counted in " +
        "the file total above. Use search_file on one of them, or narrow the pattern.");
    }
    if (entriesCapped) {
      notes.push("STOPPED: hit the " + MAX_ENTRIES + "-entry discovery budget (the walk is " +
        "shallowest-first, so the deepest directories were never reached). The search " +
        "SUCCEEDED over what it reached and more files exist beyond it — narrow with " +
        "path/depth/file_pattern, or use exec `rg` for a whole-repo sweep.");
    }
    if (timeCapped) {
      notes.push("STOPPED: spent the " + WORK_BUDGET_MS + "ms work budget (kept under the " +
        SCRIPT_DEADLINE_MS + "ms script deadline so the result comes back instead of the " +
        "whole call timing out) after " + filesScanned + " file(s), " + formatSize(bytesRead) +
        " read; the last file may have been only partly walked. The search SUCCEEDED over " +
        "what it reached and unscanned files remain — narrow with path/depth/file_pattern, " +
        "or use exec `rg` for a whole-repo sweep.");
    }
    if (outputCapped) {
      notes.push("STOPPED: the next match section would pass the " + MAX_OUTPUT_CHARS +
        "-char result budget (" + (MAX_OUTPUT_CHARS / MEMORY_CHUNK_CHARS) + " memory chunks). " +
        "Every section below is complete; further matches were not appended. Lower " +
        "context_lines or max_results, or narrow the search.");
    }
    if (!clockOk) {
      notes.push("Note: this engine served no usable clock, so the time budget was not " +
        "enforced; the entry, file-size and result budgets still were.");
    }
    var truncated = entriesCapped || timeCapped || outputCapped;

    if (results.length === 0) {
      var head;
      if (candidates === 0) {
        head = "No files to search under \"" + searchPath + "\"" +
          (filePattern ? " matching file_pattern \"" + filePattern + "\"" : "") +
          " (" + visited + " entries walked, depth " + maxDepth + ").";
      } else if (matchedFiles > 0) {
        // Found, but nothing rendered. Saying "no matches" here would be a
        // lie the notes below then contradict.
        head = matchedFiles + " file(s) CONTAIN \"" + searchPattern + "\" under " + searchPath +
          ", but no line context could be rendered for any of them — see below.";
      } else {
        head = "No matches for \"" + searchPattern + "\" in " + searchPath + " — " +
          filesScanned + " file(s) scanned, " + formatSize(bytesRead) + " read" +
          (truncated ? " (PARTIAL — see below)" : " (complete: the whole tree was scanned)") + ".";
      }
      return {
        content: notes.length > 0 ? head + "\n\n" + notes.join("\n") : head,
        success: true
      };
    }

    var header = totalMatches + " match(es) in " + matchedFiles + " file(s) for \"" +
      searchPattern + "\" under " + searchPath + " — " + filesScanned + " file(s) scanned, " +
      formatSize(bytesRead) + " read" + (truncated ? " (PARTIAL — see notes at the end)" : "");
    if (totalMatches >= maxResults) {
      notes.push("Stopped at the requested max_results of " + maxResults +
        " matches; more may exist.");
    }

    var body = header + "\n\n" + results.join("\n\n");
    if (notes.length > 0) {
      body += "\n\n" + notes.join("\n");
    }
    return { content: body, success: true };
  }
}

// Split `content` into lines and collect the indices of matching ones, in
// bounded slices. Produces exactly the array `content.split("\n")` would —
// but in interruptible steps, and it scans each slice as it lands, so a match
// near the top of a big file costs only the lines above it.
//
// `deadlineAt` is an absolute Date.now() cutoff, or null when the engine
// serves no usable clock (then the caller's size bounds are the only limit).
// Returns { lines, matches, timeCapped }; `lines` holds only what was walked.
function walkLines(content, regex, maxMatches, ctx, sliceChars, deadlineAt) {
  var lines = [];
  var matches = [];
  var scanned = 0;
  var pos = 0;
  var timeCapped = false;

  while (pos < content.length) {
    if (deadlineAt !== null && Date.now() > deadlineAt) {
      timeCapped = true;
      break;
    }

    // Extend the slice to the end of the line it lands in, so a line is never
    // cut in half across two splits.
    var end = pos + sliceChars;
    if (end < content.length) {
      var nl = content.indexOf("\n", end);
      end = nl < 0 ? content.length : nl + 1;
    } else {
      end = content.length;
    }
    var part = content.slice(pos, end);
    pos = end;

    var got = part.split("\n");
    // A slice that ended on a newline leaves a trailing "" that is not a line
    // — it is the start of the next slice's first line. At EOF it IS a line
    // (the empty segment after a trailing newline), matching split.
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
    while (scanned < lines.length && matches.length < maxMatches) {
      if (regex.test(lines[scanned])) {
        matches.push(scanned);
      }
      scanned++;
    }

    // Once we have every match we were asked for AND enough trailing lines to
    // render their context, the rest of the file cannot change the answer —
    // stop reading it.
    if (matches.length >= maxMatches &&
        lines.length > matches[matches.length - 1] + ctx) {
      break;
    }
  }
  // `"".split("\n")` is [""], but the loop above never runs for an empty file.
  // Keep the two equivalent.
  if (content.length === 0) {
    lines.push("");
  }
  // Scan anything appended but not reached — only the empty-file line above,
  // since the loop scans each slice as it lands. A no-op otherwise, and it
  // keeps "what was walked" and "what was scanned" equal on every path.
  while (scanned < lines.length && matches.length < maxMatches) {
    if (regex.test(lines[scanned])) {
      matches.push(scanned);
    }
    scanned++;
  }

  return { lines: lines, matches: matches, timeCapped: timeCapped };
}

// Is this path a file? Asked only on a path that already failed as a
// directory, so it costs one metadata call on a path the tool was going to
// give up on anyway. The stat sits in its OWN try: an error path that throws
// is worse than the error it was explaining.
function pathIsFile(path) {
  try {
    var st = Nanna.stat(path);
    return !!(st && st.is_file);
  } catch (e) {
    return false;
  }
}

// Running total of the rendered sections, including the "\n\n" that joins
// them. Recomputed rather than carried so it can never drift from `results`.
function outputChars(results) {
  var n = 0;
  for (var i = 0; i < results.length; i++) {
    n += results[i].length + 2;
  }
  return n;
}

// Join a directory path with a child name, keeping a "." root implicit so the
// reported paths stay usable verbatim in a follow-up read_file call.
function joinPath(base, name) {
  if (base === "" || base === ".") {
    return name;
  }
  return base + "/" + name;
}

function hasBinaryExt(name, exts) {
  var dot = name.lastIndexOf(".");
  if (dot <= 0 || dot >= name.length - 1) {
    return false;
  }
  return exts.indexOf(name.slice(dot + 1).toLowerCase()) >= 0;
}

function formatFileMatches(filepath, lines, matchIndices, ctx) {
  var sections = [];
  var shown = {};

  for (var mi = 0; mi < matchIndices.length; mi++) {
    var matchIdx = matchIndices[mi];
    var start = Math.max(0, matchIdx - ctx);
    var end = Math.min(lines.length - 1, matchIdx + ctx);

    if (shown[matchIdx]) continue;

    var section = [];
    for (var i = start; i <= end; i++) {
      if (shown[i]) continue;
      shown[i] = true;
      var marker = i === matchIdx ? ">" : " ";
      var lineNum = String(i + 1);
      while (lineNum.length < 4) lineNum = " " + lineNum;
      section.push(marker + lineNum + ": " + lines[i]);
    }
    sections.push(section.join("\n"));
  }

  return "=== " + filepath + " (" + matchIndices.length + " match" + (matchIndices.length > 1 ? "es" : "") + ") ===\n" + sections.join("\n  ...\n");
}

function matchGlob(name, pattern) {
  var regexStr = "^" + pattern.replace(/\./g, "\\.").replace(/\*/g, ".*").replace(/\?/g, ".") + "$";
  var regex = new RegExp(regexStr, "i");
  var basename = name.split(/[/\\]/).pop();
  return regex.test(name) || regex.test(basename);
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + "B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + "KB";
  return (bytes / (1024 * 1024)).toFixed(1) + "MB";
}
