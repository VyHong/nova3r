"""Do not select a full-run step count from a single scene or partial pair."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('step_selector',Path('scripts/eval/start_full_after_step_pilot.py'))
selector=importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def pair(name,scores):
    variants=[]
    for steps,score in zip(selector.STEPS,scores):
        record={k:0. for k in selector.METRICS}
        record.update(status='ok')
        record['occluded_apd_0.05']=score
        variants.append(dict(steps=steps,results=[record]))
    return dict(sample=name,variants=variants)


class PilotSelectionTests(unittest.TestCase):
    def test_uses_three_scene_average_instead_of_first_scene_winner(self):
        pairs=[pair('a',[.1,.8,.2]),pair('b',[.9,.1,.2]),pair('c',[.9,.1,.2])]
        winner,_=selector.select_steps(pairs)
        self.assertEqual(winner,50)
        with self.assertRaises(ValueError): selector.select_steps(pairs[:1])

    def test_incomplete_or_failed_step_comparisons_do_not_count(self):
        old=selector.PILOT
        with tempfile.TemporaryDirectory() as directory:
            selector.PILOT=Path(directory)
            out=selector.PILOT/'scannetpp/paired'
            out.mkdir(parents=True)
            full=pair('a',[.1,.2,.3])
            partial=pair('b',[.1,.2,.3]); partial['variants'].pop()
            failed=pair('c',[.1,.2,.3]); failed['variants'][1]['results'][0]['status']='error'
            try:
                for record in (full,partial,failed): (out/(record['sample']+'.json')).write_text(json.dumps(record))
                self.assertEqual([r['sample'] for r in selector.complete_pairs()],['a'])
            finally: selector.PILOT=old

    def test_historical_400_step_results_do_not_enter_selection(self):
        records=[pair(str(i),[.3,.2,.1]) for i in range(3)]
        for record in records:
            old=pair('old',[1.,1.,1.])['variants'][0]
            old['steps']=400
            record['variants'].append(old)
        winner,_=selector.select_steps(records)
        self.assertEqual(winner,50)


if __name__=='__main__': unittest.main()
