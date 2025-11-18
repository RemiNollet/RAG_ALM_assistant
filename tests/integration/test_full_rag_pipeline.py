from rag_alm_assistant.evaluation import run_evaluation

def test_pipeline_f1_score():
    f1 = run_evaluation(n_samples=10)
    assert f1 > 0.6