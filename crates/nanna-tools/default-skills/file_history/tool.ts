export default {
  name: "file_history",
  requires: ["files.history", "files.restore"],
  version: "0.1.1",
  description: "Undo file writes made in this conversation. Before write_file, edit_file or file_buffer changes a file, its previous content is saved as a numbered checkpoint. action=\"list\" shows checkpoints (newest first; pass path to see one file's); action=\"restore\" with checkpoint=<number> puts that file back exactly as the checkpoint found it (a file the write created is removed). A restore saves the current content first, so it can be undone too. Changes made through exec are not tracked.",
  parameters: {
    type: "object",
    properties: {
      action: { type: "string", enum: ["list", "restore"], description: "list or restore" },
      path: { type: "string", description: "For list: only this file's checkpoints" },
      checkpoint: { type: "integer", description: "For restore: the checkpoint number from list" },
      limit: { type: "integer", description: "For list: how many to show (default 20, max 500)" }
    },
    required: ["action"]
  },
  execute: function(input) {
    var sessionId = Nanna.sessionId();
    try {
      if (input.action === "restore") {
        var r = Nanna.service("files.restore", { session_id: sessionId, checkpoint: input.checkpoint });
        if (r.action === "removed") {
          return "Restored checkpoint " + input.checkpoint + ": " + r.path + " did not exist before that write, so it was removed. The previous content was saved as a new checkpoint.";
        }
        return "Restored checkpoint " + input.checkpoint + ": " + r.path + " is back to " + r.bytes + " bytes. The content it replaced was saved as a new checkpoint.";
      }
      if (input.action !== "list") {
        return { content: "file_history: action must be \"list\" or \"restore\" (got " + JSON.stringify(input.action) + "). Nothing was done.", success: false };
      }
      var h = Nanna.service("files.history", { session_id: sessionId, path: input.path, limit: input.limit });
      if (!h.checkpoints || h.checkpoints.length === 0) {
        return input.path
          ? "No checkpoints for " + input.path + " in this conversation."
          : "No checkpoints in this conversation yet.";
      }
      var lines = [];
      for (var i = 0; i < h.checkpoints.length; i++) {
        var c = h.checkpoints[i];
        var what = c.existed ? c.bytes + " bytes" : "did not exist yet";
        lines.push("#" + c.checkpoint + "  " + c.path + "  (" + what + ", " + c.taken_at + (c.baseline ? ", first version seen" : "") + ")");
      }
      var more = h.total > h.checkpoints.length ? "\n(" + (h.total - h.checkpoints.length) + " older not shown; raise limit)" : "";
      return "Checkpoints, newest first — each is the file as it was just BEFORE a write:\n" + lines.join("\n") + more;
    } catch (e) {
      var msg = "" + (e && e.message ? e.message : e);
      return { content: "file_history: " + msg, success: false };
    }
  }
}
