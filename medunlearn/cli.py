import argparse
import json
from pathlib import Path
import sys
from .policy import POLICY, resolve


def main():
    parser=argparse.ArgumentParser(description='Final-paper medical UE experiments; no server-specific scheduler')
    commands=parser.add_subparsers(dest='command',required=True)
    plan=commands.add_parser('plan',help='Inspect the frozen main policy without installing GPU packages')
    plan.add_argument('--dataset',choices=POLICY['datasets']);plan.add_argument('--method',choices=POLICY['methods'])
    prepare=commands.add_parser('prepare',help='Rebuild corrected splits from the documented original dataset layout')
    prepare.add_argument('--data-root',type=Path,required=True);prepare.add_argument('--workspace',type=Path,required=True)
    imported=commands.add_parser('import-manifests',help='Import an authorized local final manifest directory')
    imported.add_argument('--manifests',type=Path,required=True);imported.add_argument('--data-root',type=Path,required=True)
    imported.add_argument('--workspace',type=Path,required=True)
    run=commands.add_parser('run',help='Generate, train the matched CLEAN, train UE and evaluate one Table3 condition')
    run.add_argument('--dataset',choices=POLICY['datasets'],required=True);run.add_argument('--method',choices=POLICY['methods'],required=True)
    run.add_argument('--workspace',type=Path,required=True);run.add_argument('--python',default=sys.executable)
    run.add_argument('--generation-python');run.add_argument('--dry-run',action='store_true')
    summary=commands.add_parser('summarize',help='Recompute Table3/class-wise metrics from completed predictions')
    summary.add_argument('--workspace',type=Path,required=True);summary.add_argument('--output',type=Path)
    analyze=commands.add_parser('analyze',help='Render recall, X-ray group and confusion analyses from class aggregates')
    analyze.add_argument('--results',type=Path);analyze.add_argument('--reported',action='store_true');analyze.add_argument('--output',type=Path,required=True)
    commands.add_parser('fetch-aa',help='Fetch the pinned official UEraser source for AA')
    figure=commands.add_parser('figure',help='Run one Figure5 ratio or Figure6 AA case under the matching main protocol')
    figure.add_argument('--dataset',choices=POLICY['datasets'],required=True);figure.add_argument('--method',choices=POLICY['methods'],required=True)
    figure.add_argument('--mode',choices=['ratio','aa'],required=True);figure.add_argument('--ratio',type=int,choices=[20,40,60,80])
    figure.add_argument('--workspace',type=Path,required=True);figure.add_argument('--python',default=sys.executable);figure.add_argument('--dry-run',action='store_true')
    fs=commands.add_parser('summarize-figures',help='Collect completed Figure5/6 results without individual predictions')
    fs.add_argument('--workspace',type=Path,required=True);fs.add_argument('--output',type=Path,required=True)
    for command in ['ssim','profile','tsne']:
        artifact=commands.add_parser(command,help='Reproduce '+command+' for the selected main protocol')
        artifact.add_argument('--dataset',choices=POLICY['datasets'],required=True)
        if command!='tsne':artifact.add_argument('--method',choices=POLICY['methods'],required=True)
        if command=='ssim':artifact.add_argument('--examples',type=Path,help='Private output directory for clean/poison/magnified-difference panels')
        artifact.add_argument('--workspace',type=Path,required=True);artifact.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='plan':
        print(json.dumps(resolve(args.dataset,args.method) if args.dataset and args.method else POLICY,indent=2));return
    if args.command=='fetch-aa':
        from .upstreams import fetch_aa
        print(fetch_aa());return
    if args.command=='figure':
        from .figures import run_figure
        run_figure(args.workspace,args.dataset,args.method,args.mode,args.ratio,args.python,args.dry_run);return
    if args.command=='summarize-figures':
        from .figures import summarize_figures
        summarize_figures(args.workspace,args.output);return
    if args.command in ['ssim','profile','tsne']:
        from . import artifacts
        function=getattr(artifacts,args.command)
        if args.command=='ssim':function(args.workspace,args.dataset,args.method,args.output,args.examples)
        elif args.command=='tsne':function(args.workspace,args.dataset,args.output)
        else:function(args.workspace,args.dataset,args.method,args.output)
        return
    if args.command=='prepare':
        from .data import prepare
        print(prepare(args.data_root,args.workspace));return
    if args.command=='import-manifests':
        from .data import import_manifests
        print(import_manifests(args.manifests,args.data_root,args.workspace));return
    if args.command=='run':
        from .runtime import run
        run(args.workspace,args.dataset,args.method,args.python,args.generation_python,args.dry_run);return
    if args.command=='summarize':
        from .summarize import summarize
        summarize(args.workspace,args.output);return
    if args.command=='analyze':
        from .analysis import analyze
        from .policy import REPO
        if not args.reported and args.results is None:parser.error('Use --reported or --results')
        analyze(REPO/'reported_results' if args.reported else args.results,args.output)
