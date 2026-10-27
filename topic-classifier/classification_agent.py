import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

CATEGORIES = ('World', 'Sports', 'Business', 'Sci/Tech')
SEED = 42
PROMPT = '''You classify AG News articles into exactly one of: World, Sports, Business, Sci/Tech.
This is zero-shot classification. Choose the single best topic for the article.
Reply with ONLY a JSON object containing exactly two keys: "category" and "justification".
"category" must be one of the four names exactly. "justification" must be one brief sentence
based on the article. Do not include hidden reasoning, markdown, or extra keys.

Article:\n{article}'''


def validate_response(content):
    """Return a validated two-field dictionary or raise ValueError."""
    if not isinstance(content, str):
        raise ValueError('Expected a text response')
    try:
        data = json.loads(content.strip())
    except json.JSONDecodeError as exc:
        raise ValueError('Model response is not valid JSON') from exc
    if not isinstance(data, dict) or set(data) != {'category', 'justification'}:
        raise ValueError('JSON must contain exactly category and justification')
    if data['category'] not in CATEGORIES:
        raise ValueError(f'Invalid category: {data["category"]!r}')
    if not isinstance(data['justification'], str) or not data['justification'].strip():
        raise ValueError('justification must be a nonempty string')
    data['justification'] = data['justification'].strip()
    if len(data['justification']) > 400:
        raise ValueError('justification must be brief (400 characters maximum)')
    return data


def build_agent():
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Missing OPENAI_API_KEY in environment or .env')
    model = ChatOpenAI(
        model=os.getenv('NAUTILUS_MODEL', 'gpt-oss'),
        base_url=os.getenv('NAUTILUS_BASE_URL', 'https://ellm.nrp-nautilus.io/v1'),
        api_key=os.environ['OPENAI_API_KEY'],
        temperature=0,
        timeout=120,
        max_retries=0,
    )
    # LangChain's agent framework, following the course-provided agent_lang.py.
    return create_agent(model=model, tools=[])


def extract_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(block.get('text', '') for block in content if isinstance(block, dict) and block.get('type') == 'text')
    return str(content)


def classify(agent, article):
    """Retry one invalid response; return only validated category/justification."""
    if not isinstance(article, str) or not article.strip():
        raise ValueError('Article must be nonempty text')
    error = None
    for attempt in range(2):
        prompt = PROMPT.format(article=article)
        if attempt:
            prompt += '\n\nYour previous response failed validation. Return valid JSON only.'
        result = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
        try:
            return validate_response(extract_text(result['messages'][-1]))
        except ValueError as exc:
            error = exc
    raise ValueError(f'Invalid model output after one retry: {error}')


def sample_indices(labels, sample_size):
    import numpy as np
    if sample_size < 50 or sample_size > 7600 or sample_size % 4:
        raise ValueError('Sample size must be divisible by 4, between 50 and 7600')
    rng = np.random.default_rng(SEED)
    return sorted(int(i) for k in range(4) for i in rng.choice(np.flatnonzero(labels == k), sample_size // 4, replace=False))


def evaluate(agent, args):
    import numpy as np
    from datasets import load_dataset
    from joblib import load
    from sklearn.metrics import accuracy_score

    ds = load_dataset('fancyzhx/ag_news')['test']
    if len(ds) != 7600:
        raise ValueError('Expected official AG News test split of 7600 examples')
    labels = np.asarray(ds['label'], dtype=int)
    indices = sample_indices(labels, args.sample_size)
    texts = [ds[int(i)]['text'] for i in indices]
    truths = [CATEGORIES[int(labels[i])] for i in indices]

    # Only load trusted, locally created Part 1 joblib artifacts.
    vec_path = args.models_dir / 'best_vectorizer.joblib'
    mlp_path = args.models_dir / 'best_mlp.joblib'
    if not vec_path.exists() or not mlp_path.exists():
        raise FileNotFoundError(f'Missing Part 1 models in {args.models_dir}. Run topic_classifier.py first.')
    vec, mlp = load(vec_path), load(mlp_path)
    t0 = time.perf_counter()
    mlp_predictions = [CATEGORIES[int(k)] for k in mlp.predict(vec.transform(texts))]
    mlp_seconds = time.perf_counter() - t0

    rows = []
    start = time.perf_counter()
    for pos, (index, article, truth, mlp_pred) in enumerate(zip(indices, texts, truths, mlp_predictions), 1):
        t0 = time.perf_counter()
        try:
            answer = classify(agent, article)
            error = None
        except Exception as exc:
            answer = None
            error = f'{type(exc).__name__}: {exc}'
        latency = time.perf_counter() - t0
        rows.append({'test_index': index, 'true_category': truth, 'mlp_category': mlp_pred,
                     'agent_category': answer['category'] if answer else None,
                     'justification': answer['justification'] if answer else None,
                     'agent_latency_seconds': round(latency, 4), 'error': error})
        print(f'{pos}/{len(indices)} true={truth} mlp={mlp_pred} agent={rows[-1]["agent_category"]} latency={latency:.2f}s', flush=True)
    total = time.perf_counter() - start
    valid = [r for r in rows if r['agent_category'] is not None]
    # Invalid responses count as incorrect, rather than inflating accuracy by excluding them.
    agent_acc = sum(r['agent_category'] == r['true_category'] for r in rows) / len(rows)
    mlp_acc = accuracy_score(truths, mlp_predictions)
    summary = {
        'model': os.getenv('NAUTILUS_MODEL', 'gpt-oss'), 'seed': SEED,
        'sample_size': len(rows), 'per_category': len(rows) // 4,
        'mlp_accuracy': float(mlp_acc), 'agent_accuracy': float(agent_acc),
        'agent_valid_responses': len(valid), 'agent_failed_responses': len(rows) - len(valid),
        'average_agent_latency_seconds': sum(r['agent_latency_seconds'] for r in rows) / len(rows),
        'total_agent_evaluation_seconds': total, 'mlp_prediction_seconds': mlp_seconds,
        'agent_cost': 'Unavailable: Nautilus cluster usage/pricing not provided',
        'agent_resources': 'Nautilus-hosted gpt-oss; GPU allocation depends on deployment',
        'methodology': 'Same seed-42 stratified test examples; no test-example tuning. Failed agent calls count as incorrect.',
        'comparison': [
            {'method': 'MLP', 'accuracy': float(mlp_acc), 'latency_seconds_per_article': mlp_seconds / len(rows),
             'total_seconds': mlp_seconds, 'cost_resources': 'Local CPU/RAM; trained on 120000 labeled articles',
             'limitations': 'Requires labeled training; errors need post-hoc analysis'},
            {'method': 'gpt-oss agent', 'accuracy': float(agent_acc),
             'latency_seconds_per_article': sum(r['agent_latency_seconds'] for r in rows) / len(rows),
             'total_seconds': total, 'cost_resources': 'Nautilus GPU inference; cost unavailable',
             'limitations': 'Inference latency, service access, and possibly incorrect justifications'}
        ]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'agent_results.json').write_text(json.dumps({'summary': summary, 'examples': rows}, indent=2), encoding='utf-8')
    with (args.output_dir / 'agent_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary['comparison'][0].keys())
        writer.writeheader()
        writer.writerows(summary['comparison'])
    print('\nCOMPARISON TABLE (JSON):')
    print(json.dumps(summary['comparison'], indent=2))
    print(f'\nSaved results in {args.output_dir.resolve()}')


def main():
    parser = argparse.ArgumentParser(description='AG News zero-shot gpt-oss LangChain classification agent')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--article', help='Single news article text')
    mode.add_argument('--evaluate', action='store_true', help='Evaluate same stratified test articles as Part 1 MLP')
    parser.add_argument('--sample-size', type=int, default=200, help='Stratified sample size; use >=50 only if 200 is infeasible')
    parser.add_argument('--models-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    agent = build_agent()
    if args.article is not None:
        try:
            print(json.dumps(classify(agent, args.article), indent=2))
        except (ValueError, RuntimeError) as exc:
            parser.exit(1, f'Classification error: {exc}\n')
    else:
        evaluate(agent, args)


if __name__ == '__main__':
    main()
