export default {
  name: "read_file",
  version: "0.1.4",
  output: "memory",
  description: "Read a file from the filesystem. Returns the file contents with line numbers. Supports optional offset and limit for reading portions of large files.",
  parameters: {
    type: "object",
    properties: {
      file_path: { type: "string", description: "Path to the file to read. Relative paths are resolved against the workspace directory." },
      offset: { type: "integer", description: "Line number to start reading from (1-indexed). Default: 1" },
      limit: { type: "integer", description: "Maximum number of lines to read. Default: all lines" }
    },
    required: ["file_path"]
  },
  execute: function(input) {
    // Accept multiple parameter name variants from different models.
    // All failures are RETURNED, not thrown — a thrown error reaches the
    // model under stacked "Execution failed:" prefixes and reads as
    // corruption (observed live: an unguarded stat on a missing README).
    var filePath = input.file_path || input.filePath || input.path || input.file || input.filename;
    if (!filePath) {
      return { content: "read_file failed: missing file_path. Call it again with file_path set to the file you want to read.", success: false };
    }

    var MAX_SIZE = 10 * 1024 * 1024;

    var stat;
    try {
      stat = Nanna.stat(filePath);
    } catch (e) {
      return {
        content: "read_file: '" + filePath + "' does not exist (or is unreadable). Nothing was read. Check the path, or create the file first if you meant to write it.",
        success: false
      };
    }
    // The converse of the listing tools' file-for-directory case, and the
    // stat that answers it is already in hand — zero extra syscalls. It also
    // closes a latent uncaught throw: Nanna.readFile on a directory raises
    // OUTSIDE any try here, so it escaped as an "Execution failed:" prefix,
    // exactly what lines 17-19 above say must never happen.
    if (stat.is_dir) {
      return {
        content: "read_file: '" + filePath + "' is a DIRECTORY, not a file — use list_dir to " +
          "see its contents. Nothing was read.",
        success: false
      };
    }
    if (stat.size > MAX_SIZE) {
      return "Error: File is too large (" + (stat.size / 1024 / 1024).toFixed(1) + "MB). Maximum is 10MB. Use offset/limit to read portions.";
    }

    // Guarded like the stat above: a file that is not UTF-8 text (an image,
    // a PDF, a binary) makes the bridge raise, and an unguarded raise reaches
    // the model as a stacked "Execution failed:" script error.
    var content;
    try {
      content = Nanna.readFile(filePath);
    } catch (e) {
      return {
        content: "read_file: '" + filePath + "' could not be read as text (" + e + "). If it " +
          "is a PDF or an image, use read_pdf, ocr or describe_image. Nothing was read.",
        success: false
      };
    }

    // Read-recency mark (P22 Tier 3): record that the session saw this
    // file's content NOW. write_file/file_buffer consult these marks — a
    // shrinking whole-file rewrite of a file that CHANGED after its last
    // recorded read is held once and answered with the current content
    // (the blind-rewrite guard; design comment in write_file). A partial
    // read counts: the guard exists to catch zero-read rewrites from a
    // stale context copy, not to police how much of the file was viewed
    // (fail-open bias). Key normalization mirrors write_file's canonical
    // ledger key exactly, or the mark misses. All I/O is best-effort.
    try {
      var READMARK_STATE = ".nanna/read_marks.json";
      var READMARK_MAX_ENTRIES = 200;
      var normKey = function(path) {
        var k = path.split("\\").join("/");
        // Case folds only where the filesystem does: on Linux `README.md` and
        // `readme.md` are two files, and one key let a read of either count for both.
        if (Nanna.platform === "win32" || Nanna.platform === "darwin") k = k.toLowerCase();
        while (k.indexOf("./") === 0) k = k.substring(2);
        while (k.indexOf("//") !== -1) k = k.split("//").join("/");
        return k;
      };
      var markKey = normKey(String(filePath));
      try {
        var wd = Nanna.workdir();
        if (wd) {
          var w = normKey(String(wd));
          if (w.charAt(w.length - 1) !== "/") w += "/";
          if (markKey.indexOf(w) === 0 && markKey.length > w.length) markKey = markKey.substring(w.length);
        }
      } catch (eWd) {
        // No workdir — the raw spelling is the key, as in the ledger.
      }
      var marks;
      try {
        marks = JSON.parse(Nanna.readFile(READMARK_STATE));
        if (!marks || typeof marks !== "object" || Array.isArray(marks)) marks = {};
      } catch (eLd) {
        marks = {};
      }
      marks[markKey] = { at: Date.now() };
      var mKeys = Object.keys(marks);
      if (mKeys.length > READMARK_MAX_ENTRIES) {
        mKeys.sort(function(a, b) {
          return ((marks[a] && marks[a].at) || 0) - ((marks[b] && marks[b].at) || 0);
        });
        var evict = mKeys.length - READMARK_MAX_ENTRIES;
        for (var ev = 0; ev < evict; ev++) delete marks[mKeys[ev]];
      }
      Nanna.writeFile(READMARK_STATE, JSON.stringify(marks));
    } catch (eMark) {
      // Marks are an aid to the write guards, never a reason a read fails.
    }

    var lines = content.split("\n");
    var totalLines = lines.length;

    var startLine = Math.max(1, input.offset || 1);
    var endLine = input.limit ? Math.min(totalLines, startLine + input.limit - 1) : totalLines;

    var numbered = [];
    for (var i = startLine - 1; i < endLine; i++) {
      var padLen = String(endLine).length;
      var lineNum = String(i + 1);
      while (lineNum.length < padLen) {
        lineNum = " " + lineNum;
      }
      numbered.push(lineNum + "\t" + lines[i]);
    }

    var result = numbered.join("\n");

    if (startLine > 1 || endLine < totalLines) {
      result += "\n\n// Showing lines " + startLine + "-" + endLine + " of " + totalLines;
    }

    return result;
  }
}
