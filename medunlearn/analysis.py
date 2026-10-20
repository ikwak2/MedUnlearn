"""Publication plots and X-ray error decomposition from anonymous class aggregates."""
import json
from pathlib import Path
from .io import writecsv
from .policy import POLICY


def analyze(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    results,output=Path(results),Path(output);output.mkdir(parents=True,exist_ok=True)
    reports=json.loads((results/'aggregates.json').read_text())
    lookup={(r['dataset'],r['method']):r for r in reports};methods=POLICY['methods']
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    groups=[];distributions=[]
    for dataset in POLICY['datasets']:
        available=[r for r in reports if r['dataset']==dataset]
        if not available:continue
        classes=available[0]['classes'];support=available[0]['support']
        control=np.full((len(methods),len(classes)),np.nan);recall=control.copy()
        for i,method in enumerate(methods):
            r=lookup.get((dataset,method))
            if r is None:continue
            control[i]=100*np.diag(r['clean_confusion_matrix'])/np.array(support)
            recall[i]=100*np.diag(r['confusion_matrix'])/np.array(support)
        fig,axes=plt.subplots(1,2,figsize=(19,6.5))
        for axis,values,cmap,vmin,vmax,title in [(axes[0],control,'Purples',0,100,'Matched CLEAN recall (%)'),
                (axes[1],control-recall,'RdBu_r',-100,100,'Matched CLEAN − UE recall (pp)')]:
            color=plt.get_cmap(cmap).copy();color.set_bad('#dddddd')
            im=axis.imshow(np.ma.masked_invalid(values),cmap=color,vmin=vmin,vmax=vmax,aspect='auto')
            axis.set_title(title);axis.set_yticks(range(len(methods)),methods,fontsize=8)
            axis.set_xticks(range(len(classes)),[f'{c}\n(n={n:,})' for c,n in zip(classes,support)],rotation=50,ha='right',fontsize=7)
            for i in range(len(methods)):
                for j in range(len(classes)):
                    value=values[i,j]
                    axis.text(j,i,'—' if np.isnan(value) else f'{value:.1f}',ha='center',va='center',fontsize=6.5,
                              color='white' if np.isfinite(value) and (value>68 or value< -65) else '#222222')
            fig.colorbar(im,ax=axis,fraction=.025,pad=.015)
        fig.suptitle(dataset+' | each method uses its own matched CLEAN; gray = unfinished')
        fig.tight_layout();stem=dataset.replace('(','_').replace(')','')
        fig.savefig(output/('recall_drop_'+stem+'.pdf'),bbox_inches='tight')
        fig.savefig(output/('recall_drop_'+stem+'.png'),dpi=190,bbox_inches='tight');plt.close(fig)
        with PdfPages(output/('confusion_distribution_'+stem+'.pdf')) as pdf:
            for r in available:
                fig,axes=plt.subplots(2,2,figsize=(13,12))
                for row,(name,cm) in enumerate([('Matched CLEAN',r['clean_confusion_matrix']),('UE',r['confusion_matrix'])]):
                    cm=np.array(cm);shares=cm/np.array(support)[:,None]
                    axes[row,0].imshow(shares,cmap='Blues',vmin=0,vmax=1)
                    axes[row,0].set_title(name+' | row-normalized confusion')
                    axes[row,0].set_xticks(range(len(classes)),classes,rotation=60,ha='right',fontsize=6)
                    axes[row,0].set_yticks(range(len(classes)),classes,fontsize=6)
                    axes[row,0].set_xlabel('Predicted class');axes[row,0].set_ylabel('True class')
                    x=np.arange(len(classes));pred=cm.sum(0)
                    axes[row,1].bar(x-.2,np.array(support)/r['n']*100,.4,label='True support',color='gray')
                    axes[row,1].bar(x+.2,pred/r['n']*100,.4,label='Predictions')
                    axes[row,1].set_xticks(x,classes,rotation=60,ha='right',fontsize=6)
                    axes[row,1].set_ylim(0,105);axes[row,1].set_ylabel('Test samples (%)');axes[row,1].legend()
                fig.suptitle(dataset+' | '+r['method']);fig.tight_layout(h_pad=3,w_pad=3)
                pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
                distributions.extend(dict(dataset=dataset,method=r['method'],class_label=c,true_n=n,
                    predicted_n=int(sum(row[i] for row in r['confusion_matrix']))) for i,(c,n) in enumerate(zip(classes,support)))
                if dataset=='X-ray(fixed)':
                    nf=classes.index('No Finding');disease=[i for i in range(len(classes)) if i!=nf]
                    cm=np.array(r['confusion_matrix']);clean=np.array(r['clean_confusion_matrix'])
                    nf_drop=100*(clean[nf,nf]-cm[nf,nf])/support[nf]
                    disease_n=r['n']-support[nf]
                    disease_drop=100*sum(clean[i,i]-cm[i,i] for i in disease)/disease_n
                    groups.append(dict(method=r['method'],clean_no_finding_recall_pct=100*clean[nf,nf]/support[nf],
                        no_finding_recall_pct=100*cm[nf,nf]/support[nf],
                        clean_disease_macro_recall_pct=100*np.mean([clean[i,i]/support[i] for i in disease]),
                        disease_macro_recall_pct=100*np.mean([cm[i,i]/support[i] for i in disease]),
                        disease_weighted_recall_pct=100*sum(cm[i,i] for i in disease)/disease_n,
                        no_finding_acc_drop_contribution_pp=support[nf]/r['n']*nf_drop,
                        disease_acc_drop_contribution_pp=disease_n/r['n']*disease_drop,
                        accuracy_drop_pp=100*(np.trace(clean)-np.trace(cm))/r['n']))
    writecsv(output/'xray_no_finding_vs_disease.csv',groups)
    writecsv(output/'prediction_distribution.csv',distributions)
    print('Rendered class analyses:',output)
