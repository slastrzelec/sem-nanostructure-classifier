import pytest

from semcls.splits import canon_variant, class_names, get_split, split_vector


@pytest.fixture
def sp():
    n = 12
    return {
        "files": [f"A/{i}.jpg" for i in range(n)],
        "class": ["a", "b", "c"] * 4,
        "random": ["train", "train", "val", "test"] * 3,
        "group": {"0.94": {"cluster": list(range(n)), "split": ["test", "train", "train", "val"] * 3},
                  "0.9": {"cluster": list(range(n)), "split": ["val", "train", "train", "test"] * 3}},
    }


def test_test_part_is_blocked_by_default(sp):
    with pytest.raises(PermissionError):
        get_split(sp, "random", ("train", "val", "test"))
    assert set(get_split(sp, "random", ("train", "val"))) == {"train", "val"}


def test_test_part_available_when_explicit(sp):
    files, y = get_split(sp, "random", ("test",), allow_test=True)["test"]
    assert len(files) == len(y) == 3


def test_parts_are_disjoint_and_complete(sp):
    for variant in ("random", "group:0.94"):
        d = get_split(sp, variant, ("train", "val", "test"), allow_test=True)
        names = [f for p in d.values() for f in p[0]]
        assert sorted(names) == sorted(sp["files"])


def test_unknown_variant_and_classes(sp):
    with pytest.raises(KeyError):
        split_vector(sp, "group:0.5")
    assert class_names(sp) == ["a", "b", "c"]


def test_threshold_spelling_does_not_matter(sp):
    assert canon_variant("group:0.90") == canon_variant("group:0.9") == "group:0.9"
    assert split_vector(sp, "group:0.90") == split_vector(sp, "group:0.9")
    assert canon_variant("random") == "random"
