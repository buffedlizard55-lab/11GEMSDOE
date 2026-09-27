import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "fetch_leaderboard", Path(__file__).parents[1] / "scripts" / "fetch_leaderboard.py"
)
fetch_leaderboard = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(fetch_leaderboard)


FIXTURE = """
<html><body><table>
<thead><tr><th>Rank</th><th>Team members</th><th>Participant</th><th>Best public DW-Tversky</th><th>Shared work</th></tr></thead>
<tbody>
<tr><td>#1</td><td></td><td><a href="/users/DARD/">DARD</a><br>4d ago<br>10 submissions</td><td>0.3049</td><td></td></tr>
<tr><td>#2</td><td></td><td><a href="/users/alex/">alex</a><br>2d ago<br>4 submissions</td><td>0.2993</td><td></td></tr>
<tr><td>#3</td><td></td><td><a href="/users/third/">third</a></td><td>0.2854</td><td></td></tr>
<tr><td>#4</td><td></td><td><a href="/users/fourth/">fourth</a></td><td>0.2843</td><td></td></tr>
<tr><td>#5</td><td></td><td><a href="/users/fifth/">fifth</a></td><td>0.2806</td><td></td></tr>
</tbody></table></body></html>
"""


class LeaderboardParserTests(unittest.TestCase):
    def test_parses_public_scores_and_participants(self):
        result = fetch_leaderboard.parse_leaderboard(FIXTURE, fetched_at="2026-09-27T00:00:00Z")
        self.assertEqual(result["row_count"], 5)
        self.assertEqual(result["leader"], {"rank": 1, "participant": "DARD", "score": "0.3049"})
        self.assertEqual(result["rows"][1]["participant"], "alex")
        self.assertEqual(result["rows"][1]["score"], "0.2993")

    def test_rejects_login_or_unexpected_page(self):
        with self.assertRaises(ValueError):
            fetch_leaderboard.parse_leaderboard("<html><form>Log in</form></html>")

    def test_requires_at_least_five_scored_rows(self):
        html = FIXTURE.replace("<tr><td>#5</td><td></td><td><a href=\"/users/fifth/\">fifth</a></td><td>0.2806</td><td></td></tr>", "")
        with self.assertRaises(ValueError):
            fetch_leaderboard.parse_leaderboard(html)

    def test_rejects_nonofficial_source_without_request(self):
        with mock.patch.object(fetch_leaderboard, "urlopen") as urlopen:
            with self.assertRaisesRegex(ValueError, "only the HTTPS public DrivenData"):
                fetch_leaderboard.fetch("file:///etc/passwd")
            urlopen.assert_not_called()

    def test_failed_refresh_preserves_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "leaderboard.json"
            output.write_text("known-good snapshot", encoding="utf-8")
            error_output = io.StringIO()
            with mock.patch.object(fetch_leaderboard, "fetch", side_effect=ValueError("login page")):
                with contextlib.redirect_stderr(error_output):
                    result = fetch_leaderboard.main(["--output", str(output)])
            self.assertEqual(result, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "known-good snapshot")
            self.assertIn("not updated", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
