import pytest
from src.preprocessing.prepare import normalize, run

def test_normalize():
    cols={k:k for k in ('ticket_id','subject','description','category','priority','support_group')}
    row={k:' x  y ' for k in cols}
    assert normalize(row,cols,'synthetic')['subject']=='x y'

def test_missing_column():
    with pytest.raises(ValueError): normalize({'ticket_id':'1'}, {'ticket_id':'ticket_id'}, 'test')

def test_disabled_sources(tmp_path):
    cfg=tmp_path/'config.yaml';cfg.write_text('sources: []',encoding='utf-8')
    with pytest.raises(ValueError,match='No hay fuentes'): run(cfg,tmp_path/'out.jsonl')
