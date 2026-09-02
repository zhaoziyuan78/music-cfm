import zipfile
from pathlib import Path

import pandas as pd

from cfmusic.data.adapters.emopia import EMOPIAAdapter
from cfmusic.data.adapters.vgmidi import VGMIDIAdapter, quadrant


def test_vgmidi_quadrants() -> None:
    assert [quadrant(*pair) for pair in [(1, 1), (-1, 1), (-1, -1), (1, -1)]] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]


def test_vgmidi_adapter_uses_piece_group(tmp_path: Path, tiny_midi_path: Path) -> None:
    phrase_dir = tmp_path / "labelled/phrases"
    phrase_dir.mkdir(parents=True)
    midi_path = phrase_dir / "phrase.mid"
    midi_path.write_bytes(tiny_midi_path.read_bytes())
    pd.DataFrame(
        [
            {
                "id": "1",
                "series": "s",
                "console": "c",
                "game": "g",
                "piece": "p",
                "midi": "phrase.mid",
                "valence": 1,
                "arousal": -1,
            }
        ]
    ).to_csv(tmp_path / "vgmidi_labelled.csv", index=False)
    record = next(iter(VGMIDIAdapter(tmp_path).discover()))
    assert record.group_id == "s::g::p"
    assert record.labels["style"] == "Q4"


def test_vgmidi_adapter_extracts_and_uses_official_phrase_archive(
    tmp_path: Path, tiny_midi_path: Path
) -> None:
    labelled_dir = tmp_path / "labelled"
    labelled_dir.mkdir(parents=True)
    archive = labelled_dir / "phrases.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("phrases/game_piece_0.mid", tiny_midi_path.read_bytes())
        bundle.writestr("__MACOSX/phrases/._game_piece_0.mid", b"\x00\xe4metadata")
    pd.DataFrame(
        [
            {
                "id": "7",
                "series": "s",
                "console": "c",
                "game": "g",
                "piece": "p",
                "midi": "labelled/phrases/game_piece_0.mid",
                "valence": -1,
                "arousal": 1,
            }
        ]
    ).to_csv(tmp_path / "vgmidi_labelled.csv", index=False)

    record = next(iter(VGMIDIAdapter(tmp_path).discover()))

    assert record.source_path == labelled_dir / "phrases/game_piece_0.mid"
    assert record.item_id == "game_piece_0"
    assert record.labels["style"] == "Q2"


def test_emopia_adapter_uses_label_csv_and_ignores_appledouble_json(
    tmp_path: Path, tiny_midi_path: Path
) -> None:
    dataset_dir = tmp_path / "EMOPIA_2.2"
    midi_dir = dataset_dir / "midis"
    midi_dir.mkdir(parents=True)
    midi_path = midi_dir / "Q3_video_id_with_underscore_2.mid"
    midi_path.write_bytes(tiny_midi_path.read_bytes())
    pd.DataFrame([{"ID": midi_path.stem, "4Q": 3, "annotator": "D"}]).to_csv(
        dataset_dir / "label.csv", index=False
    )
    resource_dir = tmp_path / "__MACOSX/EMOPIA_2.2"
    resource_dir.mkdir(parents=True)
    (resource_dir / "._timestamps.json").write_bytes(b"\x00\xe4not-json")

    record = next(iter(EMOPIAAdapter(tmp_path).discover()))

    assert record.source_path == midi_path
    assert record.group_id == "video_id_with_underscore"
    assert record.labels["style"] == "Q3"
