export default {
  name: "read_pdf",
  requires: ["pdf.read"],
  version: "0.1.0",
  output: "memory",
  description: "Extract text from a PDF file. Returns the text content of the document.",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string", description: "Path to the PDF file" },
      pages: { type: "string", description: "Page range to extract (e.g. '1-5', '3'). Default: all pages" }
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
      var result = Nanna.service("pdf.read", {
        path: file,
        pages: input.pages
      });
      var text = result.text || "(empty document)";
      var pageInfo = result.page_count ? " (" + result.page_count + " pages)" : "";
      var out = "PDF: " + input.path + pageInfo + "\n\n" + text;

      // Pages with no extractable text are reported as one of four named
      // outcomes. Saying which one happened is the difference between "this
      // document is blank" and "this is a scan nobody read for you" — they look
      // identical in the text field.
      if (result.ocr === "ran") {
        if (result.ocr_text && result.ocr_text.trim()) {
          out += "\n\n## Recovered by OCR (" + (result.ocr_images || 0) + " image(s))\n\n" +
            result.ocr_text;
        } else {
          out += "\n\n*OCR ran over " + (result.ocr_images || 0) +
            " embedded image(s) and found no text.*";
        }
      } else if (result.ocr === "unavailable" || result.ocr === "no_images") {
        out += "\n\n*" + (result.ocr_note || "Some pages had no extractable text.") + "*";
      }
      return out;
    } catch (e) {
      return "Error: PDF reading service not available. " + e;
    }
  }
}
