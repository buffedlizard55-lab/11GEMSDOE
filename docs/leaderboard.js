(() => {
  const section = document.querySelector("#leaderboard-feed");
  if (!section) return;
  const tableBody = section.querySelector("tbody");
  const updated = section.querySelector("[data-updated]");
  const completeness = section.querySelector("[data-completeness]");
  const status = section.querySelector("[data-status]");
  const leaderTargets = document.querySelectorAll("[data-current-leader]");

  fetch(`leaderboard.json?ts=${Date.now()}`, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`snapshot request returned ${response.status}`);
      return response.json();
    })
    .then((snapshot) => {
      if (!Array.isArray(snapshot.rows) || snapshot.rows.length === 0 || !snapshot.leader) {
        throw new Error("snapshot has no scored rows");
      }
      for (const target of leaderTargets) target.textContent = snapshot.leader.score;
      updated.textContent = snapshot.fetched_at_utc || "unknown";
      completeness.textContent = snapshot.snapshot_complete
        ? `${snapshot.row_count} scored rows · refreshed automatically every six hours`
        : `${snapshot.row_count} top rows · fallback snapshot; scheduled refresh will replace it`;
      const rows = snapshot.rows.slice(0, 10);
      tableBody.replaceChildren(...rows.map((row) => {
        const tr = document.createElement("tr");
        for (const value of [row.rank, row.participant, row.score]) {
          const td = document.createElement("td");
          td.textContent = String(value);
          tr.appendChild(td);
        }
        return tr;
      }));
      status.textContent = "Public standings loaded";
      status.dataset.ready = "true";
    })
    .catch((error) => {
      status.textContent = `Showing saved snapshot; refresh unavailable (${error.message})`;
      status.dataset.ready = "false";
    });
})();
