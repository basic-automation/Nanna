export default {
  name: "find_files",
  version: "0.1.1",
  output: "memory",
  description: "Find files by NAME using a glob pattern — the 'where is it?' tool, when you know what a file is called but not where it lives. Use this instead of list_dir when you would otherwise walk a tree by hand, and instead of code_search when you are looking for a filename rather than for text inside files. Two pattern shapes: a pattern with NO slash matches the file's NAME at any depth (`*.rs` finds every Rust file in the tree; `Cargo.toml` finds every manifest), while a pattern WITH a slash is matched against the path relative to the search root (`src/**/*.rs`, `crates/*/Cargo.toml`). `*` matches within one path segment, `**` crosses segments, `?` matches one character. Returns paths relative to the search root, with sizes.",
  parameters: {
    type: "object",
    properties: {
      pattern: { type: "string", description: "REQUIRED. Glob pattern. No slash = match the file name at any depth (`*.rs`). With a slash = match the path from the search root (`src/**/*.rs`)." },
      path: { type: "string", description: "Directory to search under. Relative paths resolve against the workspace. Default: the workspace root." },
      max_results: { type: "integer", description: "Maximum matches to return. Default: 200." }
    },
    required: ["pattern"]
  },
  execute: function (input) {
    var pattern = input.pattern || input.glob || input.name || input.query;
    if (!pattern) {
      // Structured, not thrown: a thrown script error reaches the model under
      // an "Execution failed:" prefix, which reads as breakage rather than as
      // a fixable call.
      return {
        content: "find_files: missing required parameter 'pattern'. Nothing was searched. " +
          "Call again with pattern set to a glob, e.g. \"*.rs\" for every Rust file or " +
          "\"src/**/*.ts\" for TypeScript under src/.",
        success: false
      };
    }
    pattern = String(pattern);
    var root = String(input.path || input.dir || input.directory || input.root || ".");

    // -----------------------------------------------------------------------
    // Budgets. Derived, not picked.
    //
    // OUTPUT — this tool declares output:"memory", so its result is split into
    // MEMORY_CHUNK_MAX_CHARS (3200-char) chunks and every chunk becomes its own
    // memory row: one embedding, one similarity search, one write. The ceiling
    // is the same 10 chunks the other memory-routed listing tools use; a file
    // search has no reason to cost more memory writes than list_dir does.
    var MEMORY_CHUNK_CHARS = 3200;
    var MAX_OUTPUT_CHARS = 10 * MEMORY_CHUNK_CHARS;

    // SCAN — every entry crosses the bridge as its own JS object before this
    // script runs a line, which is what blew the 30s deadline in the `explore`
    // failure. Unlike list_dir, a glob is a FILTER: it must see far more
    // entries than it prints, so the scan bound cannot be derived from the
    // output budget. It is derived from marshalling cost instead — an entry
    // object is roughly 120 bytes, so 20000 entries is ~2.4 MB, the same order
    // as one large file read these tools already do. The bridge's own walk
    // prunes node_modules/.git/target/... and stops at depth 10, which keeps a
    // real workspace well under this.
    var SCAN_MAX = 20000;

    // A match line is at minimum a 1-char path, " (", "0B", ")" and a newline.
    var MIN_LINE_CHARS = 1 + 2 + 2 + 1 + 1;
    var HARD_RESULT_MAX = Math.floor(MAX_OUTPUT_CHARS / MIN_LINE_CHARS);
    var maxResults = Math.floor(Number(input.max_results || input.maxResults || 200));
    if (!(maxResults > 0)) maxResults = 200;
    if (maxResults > HARD_RESULT_MAX) maxResults = HARD_RESULT_MAX;

    var matcher = globToRegExp(pattern);
    if (!matcher) {
      return {
        content: "find_files: could not read \"" + pattern + "\" as a glob. Nothing was searched. " +
          "Supported: * (within one path segment), ** (across segments), ? (one character), " +
          "and [abc] character classes.",
        success: false
      };
    }
    // A pattern with no slash is a NAME pattern and is matched against each
    // file's base name at any depth; one with a slash is anchored to the path
    // relative to the search root. Stated in the answer too — a search whose
    // scope the caller guessed wrong reads as "the file is not there".
    var matchOnName = pattern.indexOf("/") === -1;

    var entries;
    try {
      // One over the bound: an overflow-length return proves more exist.
      entries = Nanna.listDir(root, true, SCAN_MAX + 1);
    } catch (e) {
      if (pathIsFile(root)) {
        return {
          content: "find_files: \"" + root + "\" is a FILE, not a directory — `path` is the " +
            "directory to search UNDER. Nothing was searched. Pass its parent directory, or " +
            "use search_file to look inside that one file.",
          success: false
        };
      }
      return {
        content: "find_files: could not read \"" + root + "\" (" + e + "). Nothing was searched.",
        success: false
      };
    }

    // A recursive walk over a FILE does not throw — the bridge yields the file
    // itself at depth 0 and then skips the root, so the listing comes back
    // EMPTY. Without this the answer is "no file matched", which is a flat
    // falsehood about a wrong-kind-of-path: it tells the caller the tree holds
    // nothing when the tree was never searched. Same one stat as the catch
    // above, and it only runs when a listing came back with nothing in it.
    if (entries.length === 0 && pathIsFile(root)) {
      return {
        content: "find_files: \"" + root + "\" is a FILE, not a directory — `path` is the " +
          "directory to search UNDER. Nothing was searched. Pass its parent directory, or " +
          "use search_file to look inside that one file.",
        success: false
      };
    }

    var scanCapped = entries.length > SCAN_MAX;
    if (scanCapped) entries.length = SCAN_MAX;

    // Strip the root the BRIDGE resolved, not the raw argument: listDir
    // names entries by their resolved absolute path (`/ws/./src/main.rs` for
    // the default "."), so a relative or default root never prefixed them,
    // every "relative" path stayed absolute, and a pattern with a slash
    // (`src/**/*.rs`) could not match anything.
    var resolvedRoot = root;
    try {
      var rootStat = Nanna.stat(root);
      if (rootStat && rootStat.path) resolvedRoot = String(rootStat.path);
    } catch (eRoot) {
      // Listing worked, so the root exists; keep the argument as written.
    }
    var rootPrefix = dropDotSegments(normalizeSlashes(resolvedRoot));
    if (rootPrefix.charAt(rootPrefix.length - 1) !== "/") rootPrefix += "/";

    var matches = [];
    var filesSeen = 0;
    for (var i = 0; i < entries.length; i++) {
      var entry = entries[i];
      if (entry.entry_type !== "file") continue;
      filesSeen++;
      var absolute = dropDotSegments(normalizeSlashes(String(entry.name)));
      var relative = absolute.indexOf(rootPrefix) === 0
        ? absolute.substring(rootPrefix.length)
        : absolute;
      var subject = matchOnName ? baseName(relative) : relative;
      if (!matcher.test(subject)) continue;
      matches.push({ path: relative, size: entry.size });
      // Keep scanning past the cap ONLY far enough to know the answer is
      // truncated — one extra match proves it without collecting a whole tree.
      if (matches.length > maxResults) break;
    }

    var resultsCapped = matches.length > maxResults;
    if (resultsCapped) matches.length = maxResults;

    matches.sort(function (a, b) {
      if (a.path < b.path) return -1;
      if (a.path > b.path) return 1;
      return 0;
    });

    var notes = [];
    // The bridge prunes and depth-limits on its own. Silent omission reads as
    // "the tree is smaller than it is", so it is stated rather than discovered.
    notes.push("Searched " + filesSeen + " files under \"" + root + "\" (" +
      (matchOnName
        ? "pattern has no slash, so it matched file NAMES at any depth"
        : "pattern has a slash, so it matched paths relative to the search root") +
      "). node_modules, .git, target, dist, build, .venv, venv, __pycache__, .next, .nuxt and " +
      ".cache are never descended into, and the walk stops at 10 levels deep.");

    if (scanCapped) {
      // The load-bearing sentence. A truncated scan means "no matches" is not
      // a fact about the tree, and a caller who reads it as one stops looking.
      notes.push("SCAN TRUNCATED at " + SCAN_MAX + " entries — this tree is larger than one " +
        "search can walk, so these results are NOT COMPLETE and an absent file may simply be " +
        "past the cap. Narrow `path` to a subdirectory and search again before concluding a " +
        "file does not exist.");
    }
    if (resultsCapped) {
      notes.push("RESULTS TRUNCATED: showing the first " + maxResults + " matches in path order; " +
        "more matched. The search SUCCEEDED — narrow the pattern or raise max_results.");
    }

    if (matches.length === 0) {
      var advice = matchOnName
        ? "The pattern matched no file NAME. Check the extension, or search a parent directory."
        : "The pattern was anchored to the path from \"" + root + "\". If you meant \"anywhere " +
          "in the tree\", drop the directory part (\"" + baseName(pattern) + "\") or lead with \"**/\".";
      return {
        content: "find_files: no file matched \"" + pattern + "\". " + advice + "\n\n" +
          notes.join("\n"),
        // Not an error: a search that ran correctly and found nothing SUCCEEDED.
        // Reporting success:false here would read as a broken tool and earn a
        // retry loop instead of a different question.
        success: true
      };
    }

    var lines = [];
    var chars = 0;
    var outputCapped = false;
    for (var m = 0; m < matches.length; m++) {
      var line = matches[m].path + " (" + formatSize(matches[m].size) + ")";
      if (chars + line.length + 1 > MAX_OUTPUT_CHARS) {
        outputCapped = true;
        break;
      }
      lines.push(line);
      chars += line.length + 1;
    }
    if (outputCapped) {
      notes.push("OUTPUT TRUNCATED: showing " + lines.length + " of " + matches.length +
        " matches — the rest did not fit in " + MAX_OUTPUT_CHARS + " chars (" +
        (MAX_OUTPUT_CHARS / MEMORY_CHUNK_CHARS) + " memory chunks). The search SUCCEEDED.");
    }

    return {
      content: "Found " + matches.length + " file(s) matching \"" + pattern + "\":\n" +
        lines.join("\n") + "\n\n" + notes.join("\n"),
      success: true
    };
  }
}

// Glob -> RegExp. Every regex metacharacter that is not a glob operator is
// escaped, so a pattern like `a+b.txt` is a literal and never a regex.
// Returns null for a pattern this cannot express, rather than a regex that
// silently means something else.
function globToRegExp(glob) {
  var out = "^";
  var i = 0;
  while (i < glob.length) {
    var ch = glob.charAt(i);
    if (ch === "*") {
      if (glob.charAt(i + 1) === "*") {
        // `**` crosses path separators; `**/` also matches zero directories,
        // so `**/x` finds `x` at the root as well as `a/b/x`.
        i += 2;
        if (glob.charAt(i) === "/") {
          out += "(?:.*/)?";
          i += 1;
        } else {
          out += ".*";
        }
      } else {
        out += "[^/]*";
        i += 1;
      }
    } else if (ch === "?") {
      out += "[^/]";
      i += 1;
    } else if (ch === "[") {
      var close = glob.indexOf("]", i + 1);
      if (close === -1) return null;
      var body = glob.substring(i + 1, close);
      if (body.length === 0) return null;
      // Only ! is re-spelled; the body is otherwise passed through, so a class
      // stays a class.
      out += "[" + (body.charAt(0) === "!" ? "^" + body.substring(1) : body) + "]";
      i = close + 1;
    } else {
      out += escapeRegExp(ch);
      i += 1;
    }
  }
  out += "$";
  try {
    return new RegExp(out);
  } catch (e) {
    return null;
  }
}

function escapeRegExp(ch) {
  return ".+^$()|{}\\]".indexOf(ch) === -1 ? ch : "\\" + ch;
}

// `/ws/./src` → `/ws/src`, `/ws/.` → `/ws`: the bridge joins the workdir
// with a relative argument as written, so `.` survives as a segment.
function dropDotSegments(p) {
  var out = p;
  while (out.indexOf("/./") !== -1) out = out.split("/./").join("/");
  if (out.length > 2 && out.substring(out.length - 2) === "/.") out = out.substring(0, out.length - 2);
  return out;
}

function normalizeSlashes(path) {
  // The bridge returns native separators; globs are written with "/".
  return String(path).split("\\").join("/");
}

function baseName(path) {
  var cut = path.lastIndexOf("/");
  return cut === -1 ? path : path.substring(cut + 1);
}

// Asked only on a path that already failed as a directory, so it costs one
// metadata call on a path the tool was going to give up on anyway. It sits in
// its OWN try: an error path that throws is worse than the error it explains.
function pathIsFile(path) {
  try {
    var st = Nanna.stat(path);
    return !!(st && st.is_file);
  } catch (e) {
    return false;
  }
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + "B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + "KB";
  return (bytes / (1024 * 1024)).toFixed(1) + "MB";
}
