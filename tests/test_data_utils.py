from PIL import Image

from src.data_utils import count_and_check_sizes


def test_count_and_check_sizes(tmp_path):
    """count_and_check_sizes should count only .jpg files and report
    their distinct pixel dimensions, ignoring non-image files."""
    Image.new("RGB", (100, 100)).save(tmp_path / "a.jpg")
    Image.new("RGB", (100, 100)).save(tmp_path / "b.jpg")
    Image.new("RGB", (50, 50)).save(tmp_path / "c.jpg")
    (tmp_path / "notes.txt").write_text("this should be ignored")

    count, sizes = count_and_check_sizes(tmp_path)

    assert count == 3
    assert sizes == {(100, 100), (50, 50)}
