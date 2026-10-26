import json
from aml_adapter.locomo_refined_batch import _question_groups
from aml_adapter.locomo_refined_data import CATEGORY_NAMES, add_payloads, official, prepare, selected_conversations
from aml_adapter.vector_recovery import _flatten_metadata


def test_official_text_counts():
    conversations, questions, manifest = official()
    assert len(conversations) == manifest["conversation_count"] == 10
    assert len(questions) == manifest["question_count"] == 1382
    assert sum(row["is_multi_modality"] is False for row in questions) == 861
    assert sum(row["is_multi_modality"] is True for row in questions) == 521


def test_pilot_is_balanced_and_covers_all_conversations():
    selection, path = prepare("pilot120", 42)
    assert path.is_file()
    assert len(selection["questions"]) == 120
    assert len(selection["conversation_ids"]) == 10
    assert {category: sum(str(row["category"]) == category for row in selection["questions"])
            for category in CATEGORY_NAMES} == {category: 30 for category in CATEGORY_NAMES}
    assert all(row["is_multi_modality"] is False for row in selection["questions"])


def test_adds_are_ordered_bounded_and_text_only():
    selection, _ = prepare("pilot120", 42)
    source = selected_conversations(selection)[selection["conversation_ids"][0]]
    payloads = list(add_payloads(source, "test-user"))
    assert payloads
    assert [row["request_id"] for row in payloads] == [f"test-user:{i}" for i in range(len(payloads))]
    assert all(1 <= len(row["messages"]) <= 20 for row in payloads)
    assert all(sum(len(message["content"].split()) for message in row["messages"]) <= 2000
               for row in payloads)
    assert all(message["content"].startswith("[") for row in payloads for message in row["messages"])
    serialized = json.dumps(payloads, ensure_ascii=False)
    assert "[caption]" not in serialized and "[images]" not in serialized and "[query]" not in serialized


def test_questions_are_grouped_by_conversation():
    selection = {"conversation_ids": ["a", "b"], "questions": [
        {"qa_id": "a1", "sample_id": "a"}, {"qa_id": "a2", "sample_id": "a"},
        {"qa_id": "b1", "sample_id": "b"},
    ]}
    state = {"conversations": {"a": {"status": "completed"}, "b": {"status": "completed"}},
             "questions": {"a1": {"status": "pending"}, "a2": {"status": "completed"},
                           "b1": {"status": "failed"}}}
    assert [[row["qa_id"] for row in group] for group in _question_groups(selection, state)] == [
        ["a1"], ["b1"],
    ]
    assert [[row["qa_id"] for row in group]
            for group in _question_groups(selection, state, ["b"])] == [["b1"]]


def test_vector_recovery_flattens_fulltext_metadata_for_milvus():
    assert _flatten_metadata({
        "unit_id": "u1",
        "tags": ["a"],
        "system_metadata": {"request_id": "r1", "role": "user"},
        "user_metadata": {"custom": 3},
    }) == {
        "unit_id": "u1",
        "tags": ["a"],
        "system_metadata.request_id": "r1",
        "system_metadata.role": "user",
        "user_metadata.custom": 3,
    }
