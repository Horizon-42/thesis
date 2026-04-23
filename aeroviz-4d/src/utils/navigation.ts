export function navigateWithinApp(url: string, mode: "push" | "replace" = "push"): void {
  const method = mode === "replace" ? "replaceState" : "pushState";
  window.history[method]({}, "", url);
  window.dispatchEvent(new PopStateEvent("popstate"));
}
