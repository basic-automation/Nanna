export default {
  name: "transcribe",
  requires: ["audio.transcribe"],
  version: "0.1.0",
  output: "memory",
  description: "Transcribe audio to text using a speech-to-text service.",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string", description: "Path to audio file (mp3, wav, m4a, etc.)" },
      language: { type: "string", description: "Language code (e.g. 'en'). Default: auto-detect" }
    },
    required: ["path"]
  },
  execute: function(input) {
    try {
      // Resolved and permission-checked by the bridge (a relative path is the
      // workspace's); the service itself takes the path as given.
      var file;
      try {
        file = Nanna.stat(input.path).path;
      } catch (e) {
        return "Error: cannot read " + input.path + ": " + e;
      }
      var result = Nanna.service("audio.transcribe", {
        path: file,
        language: input.language
      });
      return "Transcription:\n\n" + (result.text || "(empty)");
    } catch (e) {
      return "Error: Transcription service not available. " + e;
    }
  }
}
