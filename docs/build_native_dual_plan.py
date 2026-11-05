"""Build the documentation-only plan for native two-video single-choice tasks."""
from pathlib import Path
import json
from html import escape
from bs4 import BeautifulSoup

R = Path(__file__).resolve().parent
old = BeautifulSoup((R/'archive/pre-native-dual-20261105/experiment_plan.html').read_text(), 'html.parser')
css = old.style.string
css += '''
.main th:first-child,.main td:first-child{width:23%}.main th:nth-child(2),.main th:nth-child(3){width:10%}
.main th,.main td{padding:7px 2px}.main .band td{background:#ddd;text-align:left;font-weight:bold}
.grid{font-size:9pt;line-height:1.4}.grid td{padding:8px 5px}.grid th:first-child{width:22%}
.inputs{font-size:8.5pt;line-height:1.5}.inputs td{padding:9px 5px}.inputs th:nth-child(1){width:14%}.inputs th:nth-child(2){width:22%}.inputs th:nth-child(3){width:20%}.inputs th:nth-child(4){width:44%}
.references{font-size:9pt;line-height:1.25}.references li{margin-bottom:2.4mm}.dash li{margin-bottom:2.8mm}
.small{font-size:8.5pt}.note{font-size:9pt;margin:3mm 0}.grid th:nth-child(2){width:auto}
'''
benchrefs = [
 ['2025-08-27','Nannan Zhu et al.','CVBench: Benchmarking Cross-Video Synergies for Complex Multimodal Reasoning','2508.19542'],
 ['2025-11-10','Tianhao Peng et al.','MVU-Eval: Towards Multi-Video Understanding Evaluation for Multimodal LLMs','2511.07250'],
 ['2025-11-15','Jingyao Li et al.','CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models','2511.12263'],
 ['2026-03-24','Purui Bai et al.','MVPBench: A Multi-Video Perception Evaluation Benchmark for Multi-Modal Video Understanding','2603.22756'],
 ['2026-05-08','Sara Ghazanfari et al.','SYNCR: A Cross-Video Reasoning Benchmark with Synthetic Grounding','2605.08412'],
 ['2026-06-01','Peiwen Sun et al.','X-Stream: Exploring MLLMs as Multiplexers for Multi-Stream Understanding','2606.02482'],
]
methodrefs = json.loads((R/'archive/pre-native-dual-20261105/references.json').read_text())['methods']
refs = benchrefs + methodrefs
idx = {r[3]:i+1 for i,r in enumerate(refs)}
def link(url,label): return f'<a href="{escape(url,quote=True)}">{label}</a>'
def cite(id,label='论文'):return link('https://arxiv.org/abs/'+id,f'{label} [{idx[id]}]')
def table(headers,rows,cls='grid'):
 return '<table class="'+cls+'"><thead><tr>'+''.join('<th>'+v+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
def bullets(items):return '<ul class="dash">'+''.join('<li>'+i+'</li>' for i in items)+'</ul>'
pages=[]
def page(title,body):
 n=len(pages)+1
 pages.append(f'<section class="page" data-page="{n}"><h1>{title}</h1>{body}<footer class="footer"><span>FRO · 原生双视频单选题实验计划 · 2026-11-05 · v6</span><span>{n}</span></footer></section>')
methods=[('random','Random'),('fastv','FastV-AV'),('fastvid','FastVID-AV'),('dycoke_ttm','DyCoke-TTM-AV'),('omnizip','OmniZip'),('omniselect','OmniSelect'),('omnipack','OmniPack'),('fro','FRO (Ours)')]
benchmarks=['CrossVid','X-Stream','CVBench','MVU-Eval','MVPBench','SYNCR']
for model in ['7B','3B']:
 body=f'<h2>Table 1. Qwen2.5-Omni-{model} · Accuracy (%)</h2>'
 headers=['Method','Final<br/>Retained','FLOPs /<br/>Full','CrossVid<br/>2V-SC','X-Stream<br/>2V-AV-SC*','CVBench<br/>2V-SC','MVU-Eval<br/>2V-SC','MVPBench<br/>2V-SC','SYNCR<br/>2V-SC']
 body+='<table class="main"><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'
 body+='<tr class="full"><td>Full Tokens</td><td>100%</td><td>100%</td>'+('<td>—</td>'*6)+'</tr>'
 for band,ratio in [('A','45%'),('B','35%')]:
  body+=f'<tr class="band"><td colspan="9">Budget {band} · 匹配同样本 FRO 最终保留 {ratio} 的计算预算</td></tr>'
  for key,name in methods:
   body+=f'<tr class="{"ours" if key=="fro" else ""}"><td>{name}</td><td>{ratio if key=="fro" else "逐题校准"}</td><td>待实测</td>'+('<td>—</td>'*6)+'</tr>'
 body+='</tbody></table>'
 body+=bullets(['2V-SC：原生恰好两路输入、一道单选题；同模型、同输入匹配 FLOPs，文本 token 全保留。','每模型 17 个配置；两个模型共 34 个配置。Final Retained 按实际可用音视频 token 计。','FLOPs / Full 由各集合累计实际输入计算；保留率、耗时与显存另表报告。','* X-Stream：Instant / Backward 的双路带音频单选子集，按提问时刻截断；离线前缀适配版。','主表待全部纳入样本完成后填报；中途比较只使用所有方法共同完成的题目。'])
 page('4 Experiment',body)

meta=json.loads((R.parent/'manifests/methods.json').read_text())
body='<h2>Benchmarks</h2>'+bullets([
 cite('2511.12263','CrossVid')+'：跨视频比较、时间理解和多视角推理；保留原生双视频单选题。',
 cite('2606.02482','X-Stream')+'：多路流式理解；保留带音频的双路即时／回溯单选题，使用可见历史前缀。',
 cite('2508.19542','CVBench')+'：跨视频对象关联、事件关联与复杂推理；保留双视频单选题。',
 cite('2511.07250','MVU-Eval')+'：多视频感知、空间关系和推理；筛选双视频单选任务，进行纯视觉评测。',
 cite('2603.22756','MVPBench')+'：多视频动作、质量及关系比较；按全部必需输入筛选原生双视频单选任务。',
 cite('2605.08412','SYNCR')+'：合成场景下的跨视频空间、运动和对象推理；保留双路单选四选一任务。',
])+'<h2>Comparison Methods</h2>'
descs={
 'random':'按模态稳定随机采样，作为无评分压缩对照。',
 'fastv':'利用浅层注意力选择 token；本计划在第 5 层后剪枝。',
 'fastvid':'根据时段密度与冗余选择、合并视觉 token。',
 'dycoke_ttm':'使用时间 token 合并模块 TTM，保留各组锚点。',
 'omnizip':'利用音频引导视觉预算，并压缩时间冗余。',
 'omniselect':'结合 AudioCLIP、问题词性过滤与模态路由进行选择。',
 'omnipack':'组合前端选择合并和模型内部剪枝。',
 'fro':'利用第 2 层特征、跨路支持及时间新颖性选择 token。'}
items=['Full Tokens（'+cite('2503.20215')+' · '+link('https://github.com/QwenLM/Qwen2.5-Omni','代码')+'）：完整保留共享预处理输入。']
for key,name in methods:
 if key=='fro':src=link('../src/fro/scoring.py','本地代码')
 elif key=='random':src=cite('2608.03812','参考论文')+' · '+link(meta['omnipack']['official'],'参考代码')
 else:src=cite(meta[key]['paper'])+' · '+link(meta[key]['official'],'代码')
 items.append(f'{name}（{src}）：{descs[key]}')
body+=bullets(items);page('4.1 Experimental Settings',body)

body='<h2>统一模型、预算与生成设置</h2>'+table(['项目','本计划设置'],[
 ['模型','Qwen2.5-Omni-3B / 7B；各方法共享同一模型权重、processor 与输入缓存。'],
 ['样本单位','一道官方题目 + 恰好两路视频；一次生成一个答案。保留原视频顺序、选项顺序及必要引用标记。'],
 ['预算 A / B','FRO 最终保留 45% / 35%；其他方法按实际 token 数及辅助计算逐题校准至同模型 FRO FLOPs，偏差 ≤0.5%。'],
 ['输出与评分','提示仅输出 Answer: A；允许选项编号按官方选项表一一映射。单次生成，精确选项匹配；不使用 LLM 裁判。'],
 ['生成长度','max_new_tokens=64；do_sample=false；num_beams=1；不要求 CoT，不自动补答或重试答案。'],
 ['上下文预算','压缩前总输入 ≤32,256 token，为输出和处理器边界预留空间；统一预处理后再比较各压缩方法。'],
 ['超限处理','按相同系数缩减两路帧数／音频片段数量，沿有效时间范围均匀重采样；保留 stream ID、时间戳及必需题面信息，记录最终长度。'],
 ['图像处理','按各集合的帧数与尺寸表处理；保留宽高比，使用同一 processor 的 28-pixel 对齐；固定帧索引，不按方法重抽帧。'],
 ['音频','仅 X-Stream 主子集启用：16 kHz、单声道、原增益；按流独立输入，原时间轴对齐；其他五集主表为纯视觉。'],
 ['精度与种子','BF16；batch_size=1 题/GPU；FlashAttention 2；seed=20261031；Random 以题目 ID 稳定派生种子。'],
 ['推理调度','4×A100 PCIe 40GB；每卡一进程；就绪队列分片；模型常驻、跨题复用解码、CPU 异步预取。'],
 ['版本与结果','新协议独立目录；固定标注版本、筛选清单及输入指纹。旧双题配对实验保留，准确率与 FLOPs 分开报告。'],
])
page('Implementation Details',body)

settings=[
 {'name':'CrossVid','paper':'https://arxiv.org/html/2511.12263v2','frames_per_stream':32,'sampling':'uniform_full_valid_interval','long_edge':360,'audio':False,'basis':'Appendix C / Table 8: Qwen2.5-Omni-7B 64 total frames, equal video allocation; long edge 360. Same setting chosen for 3B.','selected_questions':None},
 {'name':'X-Stream','paper':'https://arxiv.org/html/2606.02482v1','frames_per_stream':128,'sampling':'2fps_then_uniform_cap_over_visible_prefix','max_pixels':100352,'audio':True,'audio_sample_rate':16000,'max_audio_seconds_per_stream':120,'long_audio_segments':24,'segment_seconds':5,'visible_interval':'official_start_to_query_timestamp','question_types':['Instant','Backward'],'audio_filter':'at_least_one_stream_has_decodable_audio; report both-stream-audio separately','basis':'Paper preprocessing uses 2fps; frame/pixel/audio caps are this plan; causal prefix MC accuracy is an offline adaptation.','selected_questions':None},
 {'name':'CVBench','paper':'https://arxiv.org/html/2508.19542v4','frames_per_stream':8,'sampling':'uniform_full_valid_interval','max_pixels':200704,'audio':False,'basis':'Section 4.1: 8 frames/video, 448x448; preserve aspect ratio with 448^2 pixel budget as this plan adaptation.','selected_questions':None},
 {'name':'MVU-Eval','paper':'https://arxiv.org/html/2511.07250v1','frames_per_stream':32,'sampling':'uniform_full_valid_interval','long_edge':720,'audio':False,'basis':'Appendix B.2.1: 32 frames/video, long edge 720 for Qwen series.','selected_questions':None},
 {'name':'MVPBench','paper':'https://arxiv.org/html/2603.22756v1','frames_per_stream':64,'sampling':'uniform_full_valid_interval','max_pixels':200704,'audio':False,'basis':'Paper IV-A standardizes video-role prompts; does not state a universal frame cap. 64 frames and 448^2 pixels selected here for action comparison.','selected_questions':None},
 {'name':'SYNCR','paper':'https://arxiv.org/html/2605.08412v1','frames_per_stream':128,'sampling':'all_native_frames_up_to_cap_then_uniform','max_pixels':100352,'audio':False,'basis':'Native Habitat/Kubric/CLEVRER clips have 100/60/128 frames; retain native temporal detail up to128 as this plan, not paper inference cap. Honor crop metadata.','selected_questions':None},
]
body='<h2>按评测集设置输入参数</h2>'+table(['Benchmark','帧数 / 每路','尺寸 / 音频','原文依据与本计划选择'],[
 ['CrossVid','均匀 32 帧<br/>双路共 64 帧','长边 360<br/>纯视觉','附录 C、表 8 的 Omni-7B 使用 64 总帧并均分；3B 同配。'+cite('2511.12263')],
 ['X-Stream','2 fps 候选<br/>每路上限 128 帧','max_pixels=100,352<br/>16 kHz 单声道','2 fps 来自数据预处理；128 帧与每路 120 秒音频为本计划。只在官方开始时刻到提问时刻内采样。'+cite('2606.02482')],
 ['CVBench','均匀 8 帧<br/>双路共 16 帧','max_pixels=200,704<br/>纯视觉','§4.1 原设每视频 8 帧、448×448；本计划采用等面积上限并保留宽高比。'+cite('2508.19542')],
 ['MVU-Eval','均匀 32 帧<br/>双路共 64 帧','长边 720<br/>纯视觉','附录 B.2.1 的 Qwen 系列使用每视频 32 帧、长边上限 720。'+cite('2511.07250')],
 ['MVPBench','均匀 64 帧<br/>双路共 128 帧','max_pixels=200,704<br/>纯视觉','§IV-A 未给统一抽帧数；本计划选 64 帧以覆盖细粒度动作，448² 像素兼顾质量比较。'+cite('2603.22756')],
 ['SYNCR','原生帧率<br/>每路上限 128 帧','max_pixels=100,352<br/>无音频','依据源片段 60 / 100 / 128 帧特点选定上限；按 start_sec / end_sec 裁剪，不将短片统一降到 2 fps。'+cite('2605.08412')],
],'inputs')
body+='<h2>X-Stream 时间与音频规则</h2>'+table(['项目','设置'],[
 ['子集名称','X-Stream-2V-AV-SC-Prefix；恰好两路、原生单选、Instant / Backward、至少一路有可解码音轨。两路均带音频另报子组。'],
 ['可见时间','t ≤ 官方提问时刻；保留跨流偏移与同步关系；禁止用答案证据区间缩短历史，Forward 不进入本主表。'],
 ['长音频','每路 ≤120 秒保留全部；超出后沿可见前缀均匀选 24×5 秒，保留原始时间戳；输入超限再共同缩减。'],
 ['指标','选项匹配 Accuracy；Instant / Backward 分项；解析失败率、截断率、音轨覆盖率。'],
])
page('Implementation Details',body)

body='<h2>评测范围与筛选规则</h2>'+table(['项目','纳入规则'],[
 ['原生双视频','统计所有必需输入，包括参考视频、候选视频和裁剪片段；恰好两路，不从三／四路题中删视频凑成两路。'],
 ['题型','明确的单选题、至少两个选项、唯一有效标准答案；保留原始选项与语义。开放题、多选集合题和自由排序输出不进入主表。'],
 ['任务子集','使用发布评测 split 内全部符合条件的题；多个 split 分开记录，训练／开发数据不混入主表。'],
 ['题量','六集筛选后的 N 均待标注清单统计；CrossVid 的 4,531 双视频题不能直接作为双视频单选题数。'],
 ['缺失数据','分别记录符合条件、媒体可用、下载完成与评测完成的数量；缺失媒体不得被悄悄移出总分母。'],
 ['输入身份','按 video ID、裁剪起止及流编号标识输入；同源不同片段遵守官方题目定义。保留 bbox、对象指代等必要元数据，并同步坐标变换。'],
 ['视觉集合音轨','CrossVid / CVBench / MVU-Eval / MVPBench / SYNCR 主表禁用音频；可选音视频扩展单列，不混入纯视觉结果。'],
 ['纳入清单冻结','记录官方版本／commit、题目 ID、选项、答案索引、媒体映射与 SHA256；所有模型和方法共享同一题目集合。'],
])+'<h2>评分与汇总</h2>'+table(['项目','设置'],[
 ['确定性解析','只解析唯一的 Answer: 标签或完整单选答案；允许官方编号／字母的一一映射；出现互相冲突选项时不猜测答案。'],
 ['指标分母','Accuracy = 正确题数 / 全部纳入题数；无法解析计错，同时报告解析失败、空输出、长度截断和运行失败。'],
 ['子任务','按官方类别报告准确率与题量；MVPBench 主表使用原始 Accuracy，原文归一化 proficiency 仅作单独补充指标。'],
 ['比较与区间','中途只比较共同完成题目；最终对相同题目进行配对比较，以共享视频连通组为簇 bootstrap 1,000 次，报告 95% CI。'],
 ['跨集合汇总','各集合独立报告；只有六集都完成且样本数非零时，计算六集 Accuracy 的等权平均。'],
])
page('Implementation Details',body)

body='<h2>方法配置与预算校准</h2>'+table(['Method','固定设置'],[
 ['Full Tokens / Random','Full 保留共同输入全部 token；Random 按可用模态固定种子采样。'],
 ['FastV-AV','第 5 层后剪枝；前 5 层完整计算、保留的 KV 与注意力评分计入 FLOPs。'],
 ['FastVID-AV','全局整数视觉预算按时段分配；原有排序／合并；有音频时使用固定均匀音频基线。'],
 ['DyCoke-TTM-AV','仅 TTM，保留组锚点并使用整数预算；不启用动态解码 KV 压缩。'],
 ['OmniZip','有音频时启用音频引导；纯视觉输入使用明确的视觉路径，零音频 token，记录适配分支。'],
 ['OmniSelect','有音频保留 AudioCLIP 与词性过滤；纯视觉时固定视觉路由，保留视觉选择核心；辅助调用按实际执行计成本。'],
 ['OmniPack','前端选择 + 内部保留 50%；7B 第 18 层，3B 第 26 层；ηv/a=.25/.35，λ=.20，τv/a=.10/.05，ζ=.10。'],
 ['FRO','独立第 2 层评分；bridge=.30，top-k=3，coverage=.20，novelty=.20；同模态跨路支持；选中原始嵌入送入主解码器。'],
 ['无音频统一处理','不存在的模态预算为 0；不制造静音 token。新原生题目接口与无音频分支需在正式运行前完成适配验证。'],
])+'<h2>FLOPs 与执行记录</h2>'+table(['项目','统计方式'],[
 ['目标','每题每模型先计算 FRO 45% / 35% 的成本目标，再对其他方法校准；不可达到的预算标为 infeasible，单独报告数量。'],
 ['统计范围','1 MAC=2 FLOPs；主解码器 + 额外评分前缀 + AudioCLIP + 选择／合并矩阵运算；统一标准化 64 解码步，并另记实际生成成本。'],
 ['共同编码器','压缩发生在编码后时，匹配预算不含共同编码器；端到端 FLOPs 与延迟另列并包含编码器。逐元素算子、排序等计数范围明确记录。'],
 ['聚合','FLOPs Ratio = Σ方法成本 / Σ同输入 Full Tokens 成本；按集合、模型、档位分别算。不得复用旧固定参考输入的百分比。'],
 ['吞吐与恢复','记录预处理、编码、评分、prefill、decode、峰值显存、题/小时；每卡独立输出，按题目 ID 去重，保留结果和输入指纹。'],
])
page('Implementation Details',body)

body='<div class="ref-note">Benchmarks 在前、Methods 在后；各组按首次公开日期由早到晚排列。</div>'
for title,rs in [('Benchmarks',benchrefs),('Methods',methodrefs)]:
 body+='<h2>'+title+'</h2><ol class="references">'
 for date,authors,title_,id in rs:
  body+=f'<li id="ref-{idx[id]}"><span class="refnum">[{idx[id]}]</span> {escape(authors)} {link("https://arxiv.org/abs/"+id,escape(title_))}. arXiv:{id}, {date}.</li>'
 body+='</ol>'
page('References',body)
(R/'experiment_plan.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>FRO Native Two-Video Experiment Plan</title><style>'+css+'</style></head><body>'+''.join(pages)+'</body></html>')
(R/'references.json').write_text(json.dumps({'sort':'first public arXiv submission date, ascending within group','benchmarks':benchrefs,'methods':methodrefs},ensure_ascii=False,indent=2))
parameters={'version':'v6_native_two_video_single_choice','status':'documentation_plan_not_runtime_config','models':['3B','7B'],'methods':['full_tokens']+[x[0] for x in methods],'fro_anchor_retentions':[.45,.35],'budget_mode':'per_sample_matched_flops','budget_tolerance':.005,'max_input_tokens':32256,'decode':{'do_sample':False,'num_beams':1,'max_new_tokens':64},'flops_standard_decode_steps':64,'seed':20261031,'question_protocol':'one_native_question_two_video_inputs','question_filter':'exactly_two_required_video_inputs_and_native_single_choice','datasets':settings}
(R/'native_dual_video_parameters.json').write_text(json.dumps(parameters,ensure_ascii=False,indent=2))
(R/'native_dual_video_plan_metadata.json').write_text(json.dumps({'version':parameters['version'],'benchmarks':benchmarks,'selected_question_count':None,'configurations':34,'models':['Qwen2.5-Omni-3B','Qwen2.5-Omni-7B'],'pages':8,'references':13,'date':'2026-11-05','runtime_config_changed':False,'previous_plan':'archive/pre-native-dual-20261105/experiment_plan.pdf'},ensure_ascii=False,indent=2))
print('Generated 8-page native two-video plan and documentation parameters.')
