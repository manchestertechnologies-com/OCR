"""The web page always runs the current app.js and styles.css."""
import re


def test_page_asks_for_the_current_script_and_styles(tmp_data):
    """/static is sent without cache rules, so browsers keep app.js for hours. An old app.js
    with a new page stopped every upload, silently (it read a form field the page no longer
    has). The page asks for app.js?v=<file version>: after an update that is a new address,
    which the browser has to load."""
    from fastapi.testclient import TestClient

    from examocr.api.app import WEB, app
    with TestClient(app) as c:
        r = c.get("/")
        assert r.headers["cache-control"] == "no-cache"
        for name in ("app.js", "styles.css"):
            v = re.search(rf'"/static/{re.escape(name)}\?v=(\d+)"', r.text)
            assert v, f"{name} is asked for without a version"
            assert int(v.group(1)) == (WEB / name).stat().st_mtime_ns
            assert c.get(f"/static/{name}?v={v.group(1)}").status_code == 200
