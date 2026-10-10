export default {
  name: "web_search",
  version: "0.1.1",
  output: "memory",
  description: "Search the web using Brave Search API. Returns titles, URLs, and descriptions of search results.",
  parameters: {
    type: "object",
    properties: {
      query: { type: "string", description: "Search query" },
      count: { type: "integer", description: "Number of results to return. Default: 5" }
    },
    required: ["query"]
  },
  execute: function(input) {
    var numResults = input.count || 5;

    var apiKey = Nanna.getEnv("BRAVE_API_KEY");
    if (!apiKey) {
      // Neither route the old text named is reachable from a tool call: a
      // child shell cannot mutate the daemon's live process environment, and
      // the config field has no consumer. Advice that cannot be followed reads
      // as "try again" and gets retried.
      return "web_search is unavailable in this session: no BRAVE_API_KEY is set in the daemon's environment. Nothing was searched. Use web_fetch on a known URL, or ask the user to set the key and restart the daemon.";
    }

    var url = "https://api.search.brave.com/res/v1/web/search?q=" + encodeURIComponent(input.query) + "&count=" + numResults;
    // A DNS failure or a timeout makes the bridge raise; returned, not
    // thrown, as in web_fetch — a raise reads to the model as corruption.
    var response;
    try {
      response = Nanna.fetch(url, {
        headers: {
          "Accept": "application/json",
          "Accept-Encoding": "gzip",
          "X-Subscription-Token": apiKey
        }
      });
    } catch (e) {
      return { content: "web_search: the search request failed (" + e + "). Nothing was searched; retry later or use web_fetch on a known URL.", success: false };
    }

    if (response.status !== 200) {
      return "Error: Brave Search API returned status " + response.status + ": " + response.body.substring(0, 200);
    }

    var data;
    try {
      data = JSON.parse(response.body);
    } catch (e) {
      return "Error: Failed to parse search results";
    }

    var results = (data.web && data.web.results) || [];
    if (results.length === 0) {
      return "No results found for \"" + input.query + "\"";
    }

    var formatted = [];
    for (var i = 0; i < results.length; i++) {
      var r = results[i];
      var desc = r.description || "(no description)";
      formatted.push((i + 1) + ". " + r.title + "\n   " + r.url + "\n   " + desc);
    }

    return "Search results for \"" + input.query + "\":\n\n" + formatted.join("\n\n");
  }
}
