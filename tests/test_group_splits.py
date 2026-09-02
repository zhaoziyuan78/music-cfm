from cfmusic.data.splitting import assert_disjoint_groups, grouped_stratified_split


def test_groups_never_cross_splits() -> None:
    groups = [f"group-{index // 2}" for index in range(60)]
    labels = [f"style-{index % 3}" for index in range(60)]
    splits = grouped_stratified_split(groups, labels, seed=42)
    assert set(splits) == {"train", "validation", "test"}
    assert_disjoint_groups(groups, splits)
    owners = {
        group: {
            split for candidate, split in zip(groups, splits, strict=True) if candidate == group
        }
        for group in groups
    }
    assert all(len(values) == 1 for values in owners.values())
