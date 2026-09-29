"""Execute every notebook cell in a fresh kernel with the built-in smoke budget.

A fresh temporary output directory prevents reuse of old training/evaluation.
The delivery notebook is never overwritten. Smoke validates execution, not
convergence, statistical significance or performance rankings.
"""
import argparse
import os
from pathlib import Path
import tempfile
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default='verification/smoke_executed.ipynb')
    parser.add_argument('--inprocess', action='store_true',
                        help='Run sequentially in IPython without kernel sockets')
    args=parser.parse_args()
    source=ROOT/'notebooks/Chase_Population_Robustness_Tested.ipynb'
    nb=nbformat.read(source,as_version=4)
    with tempfile.TemporaryDirectory(prefix='qcatch_smoke_') as tmp:
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            cell.outputs=[];cell.execution_count=None
            if 'qcatch-config' in cell.metadata.get('tags',[]):
                cell.source=cell.source.replace('PROFILE = "pilot"','PROFILE = "smoke"')
                cell.source=cell.source.replace('RUN_EXPERIMENTS = False','RUN_EXPERIMENTS = True')
                cell.source=cell.source.replace('OUTPUT_ROOT = Path("chase_population_runs")',
                                                'OUTPUT_ROOT = Path('+repr(tmp)+')')
        if args.inprocess:
            execute_inprocess(nb)
        else:
            NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        nbformat.write(nb,out)
        print('Executed all cells successfully:',out)


def execute_inprocess(nb):
    import sys
    import types
    import io
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from IPython.display import display, Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    module=types.ModuleType('__main__')
    module.__dict__['__builtins__']=__builtins__
    shell=InteractiveShell.instance(user_module=module,user_ns=module.__dict__)
    sys.modules['__main__']=module
    os.chdir(ROOT)
    def show(*args,**kwargs):
        for number in plt.get_fignums():
            buffer=io.BytesIO()
            plt.figure(number).savefig(buffer,format='png',bbox_inches='tight')
            display(Image(data=buffer.getvalue()));plt.close(number)
    plt.show=show
    count=0
    for i,cell in enumerate(nb.cells):
        if cell.cell_type!='code':continue
        count+=1
        print('Executing cell',i,flush=True)
        with capture_output() as captured:
            result=shell.run_cell(cell.source,store_history=True)
        cell.outputs=[];cell.execution_count=count
        for stream in ('stdout','stderr'):
            value=getattr(captured,stream)
            if value:cell.outputs.append(nbformat.v4.new_output('stream',name=stream,text=value))
        for item in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
        error=result.error_before_exec or result.error_in_exec
        if error:
            print(captured.stdout,captured.stderr)
            raise RuntimeError(f'Notebook cell {i} failed') from error
    nb.metadata['verification']={'profile':'smoke','execution':'IPython in-process; all code cells',
                                 'performance_claim':False}

if __name__=='__main__':main()
