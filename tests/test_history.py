from ui.history import list_snapshots, load_snapshot, save_snapshot


def test_save_list_and_load_roundtrip(tmp_path):
    snap = {"query": "q1", "created_at": "2026-11-03T18:00:00",
            "fact_check_results": [{"status": "CONFIRMED"}, {"status": "UNVERIFIED"}], "final_report": "# R"}
    path = save_snapshot(snap, tmp_path)
    assert load_snapshot(path)["final_report"] == "# R"
    rows = list_snapshots(tmp_path)
    assert len(rows) == 1 and rows[0]["claims"] == 2 and rows[0]["confirmed"] == 1 and rows[0]["query"] == "q1"


def test_newest_first_and_corrupt_files_skipped(tmp_path):
    (tmp_path / "20260101_000000_000000.json").write_text("{not json")
    save_snapshot({"query": "older"}, tmp_path)
    save_snapshot({"query": "newer"}, tmp_path)
    assert [r["query"] for r in list_snapshots(tmp_path)][0] == "newer"
    assert len(list_snapshots(tmp_path)) == 2


def test_missing_directory_is_empty(tmp_path):
    assert list_snapshots(tmp_path / "nope") == []


def test_delete_only_removes_files_inside_the_history_folder(tmp_path):
    from ui.history import delete_snapshot
    path = save_snapshot({"query": "x", "final_report": "r"}, tmp_path)
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}")
    assert not delete_snapshot(outside, tmp_path) and outside.exists()
    assert delete_snapshot(path, tmp_path) and not path.exists()
    assert not delete_snapshot(path, tmp_path)


def test_overwrite_and_state_in_listing(tmp_path):
    from ui.history import overwrite_snapshot
    path = save_snapshot({"query": "x", "final_report": "r", "fact_check_results": [{"status": "CONFIRMED"}]}, tmp_path)
    overwrite_snapshot(path, {"query": "x", "final_report": "r2", "fact_check_results": []})
    assert load_snapshot(path)["final_report"] == "r2"
    assert list_snapshots(tmp_path)[0]["state"] == "partial"     # no claims -> flagged, not shown as success
