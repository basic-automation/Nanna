export default {
  name: "web_fetch",
  version: "0.1.0",
  output: "memory",
  description: "Fetch a web page and extract its text content. Strips HTML tags, scripts, and styles to return readable text.",
  parameters: {
    type: "object",
    properties: {
      url: { type: "string", description: "URL to fetch" },
      max_chars: { type: "integer", description: "Maximum characters to return. Default: 50000" }
    },
    required: ["url"]
  },
  execute: function(input) {
    var maxChars = input.max_chars || 50000;

    // A refused host, a DNS failure or a timeout is an answer, not a crash:
    // thrown, it surfaced as a script error instead of a result.
    var response;
    try {
      response = Nanna.fetch(input.url);
    } catch (e) {
      return "Error: could not fetch " + input.url + ": " + e;
    }

    if (response.status >= 400) {
      return "Error: HTTP " + response.status + " fetching " + input.url;
    }

    var contentType = response.headers["content-type"] || "";
    var text;

    if (contentType.indexOf("application/json") >= 0) {
      try {
        text = JSON.stringify(JSON.parse(response.body), null, 2);
      } catch (e) {
        text = response.body;
      }
    } else if (contentType.indexOf("text/html") >= 0 || response.body.indexOf("<html") >= 0) {
      text = htmlToText(response.body);
    } else {
      text = response.body;
    }

    if (text.length > maxChars) {
      text = text.substring(0, maxChars) + "\n\n(Truncated at " + maxChars + " characters)";
    }

    return "Fetched " + input.url + " (" + response.status + "):\n\n" + text;
  }
}

function htmlToText(html) {
  var text = html.replace(/<script[\s\S]*?<\/script>/gi, "");
  text = text.replace(/<style[\s\S]*?<\/style>/gi, "");
  text = text.replace(/<noscript[\s\S]*?<\/noscript>/gi, "");
  text = text.replace(/<\/?(p|div|br|hr|h[1-6]|li|tr|blockquote|pre|section|article|header|footer|nav|main)[^>]*>/gi, "\n");
  text = text.replace(/<li[^>]*>/gi, "\n- ");
  text = text.replace(/<[^>]+>/g, "");
  text = text.replace(/&amp;/g, "&");
  text = text.replace(/&lt;/g, "<");
  text = text.replace(/&gt;/g, ">");
  text = text.replace(/&quot;/g, "\"");
  text = text.replace(/&#39;/g, "'");
  text = text.replace(/&nbsp;/g, " ");
  text = text.replace(/[ \t]+/g, " ");
  text = text.replace(/\n[ \t]+/g, "\n");
  text = text.replace(/\n{3,}/g, "\n\n");
  text = text.trim();
  return text;
}
