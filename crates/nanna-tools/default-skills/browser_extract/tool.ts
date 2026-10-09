export default {
  name: "browser_extract",
  requires: ["browser.extract"],
  version: "0.1.0",
  output: "memory",
  description: "Extract structured data from a web page using CSS selectors or by rendering JavaScript-heavy pages.",
  parameters: {
    type: "object",
    properties: {
      url: { type: "string", description: "URL of the page" },
      selector: { type: "string", description: "CSS selector to extract. Default: 'body'" },
      attribute: { type: "string", description: "Attribute to extract (e.g. 'href', 'src'). Default: text content" }
    },
    required: ["url"]
  },
  execute: function(input) {
    try {
      var result = Nanna.service("browser.extract", {
        url: input.url,
        selector: input.selector || "body",
        attribute: input.attribute
      });
      return result.text || result.content || "(no content extracted)";
    } catch (e) {
      return { content: "Error: " + e, success: false };
    }
  }
}
