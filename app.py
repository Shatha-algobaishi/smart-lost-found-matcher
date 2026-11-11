import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import gradio as gr

SAVE_TO_DRIVE = False

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/SmartLostFoundAI')
else:
    ROOT = Path(os.environ.get('LOST_FOUND_DATA_DIR', Path(__file__).parent / 'data'))

ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / 'lost_found.db'
VECTOR_PATH = ROOT / 'vectors'

SEED_REPORTS = [
    ('lost','Black wireless headphones','Airport','Black Bluetooth headphones in a small black case near departure gate 12','Ahmed','open',2),
    ('found','Black wireless headphones','Airport','Black wireless headphones inside a black carrying case found at airport information desk','Noura','open',1),
    ('lost','Silver wristwatch','Mall','Silver wrist watch with a dark leather strap missing near the food court','Fahad','open',4),
    ('found','Silver wristwatch','Mall','Silver watch with dark brown leather strap found at the shopping mall food court','Sara','open',3),
    ('lost','Blue backpack','University','Navy blue backpack with notebooks and a laptop charger near the library','Khalid','open',5),
    ('found','Blue backpack','University','Blue school backpack containing notebooks found next to the university library','Lina','open',4),
    ('lost','Car keys','Airport','Toyota car key with blue keychain lost near airport parking area','Maha','open',7),
    ('found','Car keys','Airport','Vehicle keys with a blue key ring handed to airport security','Omar','open',6),
    ('lost','Phone','Mall','Black mobile phone with cracked screen missing near electronics shop','Hassan','open',1),
    ('found','Phone','Mall','Black smartphone with a cracked screen found in shopping center','Reem','open',1),
    ('lost','Wallet','Park','Brown leather wallet with transit card lost beside the fountain','Ali','open',3),
    ('found','Wallet','Park','Brown wallet found near the park fountain','Yara','open',2),
    ('lost','Glasses','University','Black prescription eyeglasses in a grey case missing from lecture room','Majed','open',8),
    ('found','Glasses','University','Prescription glasses with black frame found in lecture hall','Dana','open',7),
    ('lost','Laptop','Cafe','Silver laptop in dark sleeve left at cafe by the window','Tariq','open',5),
    ('found','Laptop','Cafe','Silver notebook computer in black sleeve returned to cafe counter','Hind','open',4),
    ('lost','Earbuds','Train Station','White wireless earbuds without charging case lost at platform two','Abdullah','open',2),
    ('found','Earbuds','Train Station','White earbud pair found on train platform near bench','Huda','open',2),
    ('lost','Umbrella','Mall','Small red folding umbrella left outside a fashion store','Maryam','open',6),
    ('found','Umbrella','Mall','Red compact umbrella found near mall main entrance','Saad','open',5),
    ('lost','Water bottle','Gym','Green steel water bottle with sticker misplaced at gym','Rakan','open',4),
    ('found','Water bottle','Gym','Green reusable bottle found beside treadmill','Abeer','open',3),
    ('lost','Camera','Park','Compact digital camera in blue case missing by picnic area','Bader','open',11),
    ('found','Camera','Park','Small digital camera inside blue pouch found at park','Asma','open',10),
    ('lost','Book','University','Arabic mathematics textbook with handwritten notes missing after class','Saleh','resolved',9),
    ('found','Book','University','Math textbook turned in to campus lost property','Rima','resolved',8),
    ('lost','Jacket','Airport','Grey winter jacket forgotten in waiting lounge','Yousef','open',3),
    ('found','Jacket','Airport','Grey coat found near boarding gate seating area','Amal','open',2),
]

def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

with db_connect() as con:
    con.execute("""CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY,
        report_type TEXT NOT NULL,
        item TEXT NOT NULL,
        location TEXT NOT NULL,
        description TEXT NOT NULL,
        reporter TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    n = con.execute('SELECT COUNT(*) FROM reports').fetchone()[0]
    if n == 0:
        now = datetime.now()
        for i, (t,item,location,description,reporter,status,days_ago) in enumerate(SEED_REPORTS,1):
            con.execute('INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (f'LF-{i:04d}', t, item, location, description, reporter, status,
                 (now - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M')))
    con.commit()

print('✅ SQLite ready:', DB_PATH)
print('✅ Sample reports:', len(SEED_REPORTS))

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
embedding_model = SentenceTransformer(MODEL_NAME)
vector_client = chromadb.PersistentClient(path=str(VECTOR_PATH))
collection = vector_client.get_or_create_collection(
    name='lost_found_reports',
    metadata={'hnsw:space': 'cosine'}
)

def read_reports(report_type=None, status=None):
    query = 'SELECT * FROM reports WHERE 1=1'
    params = []
    if report_type and report_type != 'all':
        query += ' AND report_type = ?'
        params.append(report_type)
    if status and status != 'all':
        query += ' AND status = ?'
        params.append(status)
    query += ' ORDER BY created_at DESC, id DESC'
    with db_connect() as con:
        rows = con.execute(query, params).fetchall()
    return [dict(row) for row in rows]

def searchable_text(row):
    return f"{row['item']}. {row['description']}. Location: {row['location']}"

def sync_index():
    rows = read_reports()
    if not rows:
        return 0
    vectors = embedding_model.encode([searchable_text(row) for row in rows], normalize_embeddings=True)
    collection.upsert(
        ids=[r['id'] for r in rows],
        embeddings=vectors.tolist(),
        documents=[searchable_text(r) for r in rows],
        metadatas=[{'report_type':r['report_type'], 'location':r['location'], 'status':r['status']} for r in rows]
    )
    return len(rows)

def semantic_search(query, report_type='all', top_k=5, only_open=True):
    rows = read_reports()
    if not rows or not str(query).strip():
        return []
    where_parts = []
    if report_type != 'all':
        where_parts.append({'report_type':report_type})
    if only_open:
        where_parts.append({'status':'open'})
    where = None
    if len(where_parts) == 1:
        where = where_parts[0]
    elif len(where_parts) > 1:
        where = {'$and':where_parts}
    candidates = [r for r in rows if (report_type == 'all' or r['report_type']==report_type) and (not only_open or r['status']=='open')]
    if not candidates:
        return []
    query_embedding = embedding_model.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(int(top_k),len(candidates)),
        where=where,
        include=['distances','metadatas','documents']
    )
    by_id = {r['id']:r for r in rows}
    matches=[]
    for rid, distance in zip(result['ids'][0], result['distances'][0]):
        if rid in by_id:
            matches.append({**by_id[rid], 'score':max(-1.0,min(1.0,1.0-float(distance)))})
    return matches

def find_matches(report_id, top_k=5):
    rows = read_reports()
    selected = next((r for r in rows if r['id']==report_id),None)
    if not selected:
        return []
    opposite = 'found' if selected['report_type']=='lost' else 'lost'
    matches = semantic_search(searchable_text(selected), opposite, max(top_k*4,12), only_open=True)
    matches = [r for r in matches if r['id']!=report_id]
    matches.sort(key=lambda r: (r['score'] + (0.06 if r['location'].casefold()==selected['location'].casefold() else 0)), reverse=True)
    return matches[:top_k]

print('✅ Vector database:', sync_index(), 'reports indexed')
print('✅ Embedding model:', MODEL_NAME)

import numpy as np

TABLE_COLS = ['رقم البلاغ','النوع','الغرض','الموقع','الوصف','الحالة','التاريخ']

def report_table(report_type='all', status='all'):
    rows = read_reports(report_type, status)
    result = [[r['id'], 'مفقود' if r['report_type']=='lost' else 'موجود',r['item'],r['location'],r['description'],'مفتوح' if r['status']=='open' else 'مغلق',r['created_at']] for r in rows]
    return pd.DataFrame(result,columns=TABLE_COLS)

def overview():
    rows = read_reports()
    total = len(rows)
    lost = sum(r['report_type']=='lost' for r in rows)
    found = sum(r['report_type']=='found' for r in rows)
    resolved = sum(r['status']=='resolved' for r in rows)
    headline = f'### 📊 إجمالي البلاغات: {total}   |   🔎 مفقودات: {lost}   |   📦 موجودات: {found}   |   ✅ مغلقة: {resolved}'
    df = pd.DataFrame(rows)
    if df.empty:
        return headline, None, None, report_table()
    by_location = df.groupby(['location','report_type']).size().reset_index(name='count')
    location_fig = px.bar(by_location,x='location',y='count',color='report_type',barmode='group',title='بلاغات المفقودات والموجودات حسب الموقع',labels={'location':'الموقع','count':'عدد البلاغات','report_type':'النوع'})
    location_fig.update_layout(legend_title_text='نوع البلاغ', margin=dict(l=10,r=10,t=55,b=10))
    by_type = df.groupby('report_type').size().reset_index(name='count')
    pie_fig = px.pie(by_type,names='report_type',values='count',hole=.5,title='توزيع أنواع البلاغات')
    pie_fig.update_layout(margin=dict(l=10,r=10,t=55,b=10))
    return headline, location_fig, pie_fig, report_table()

def new_report(report_type,item,location,description,reporter):
    values = [str(v).strip() for v in [item,location,description,reporter]]
    if not all(values):
        return '⚠️ أكمل جميع الحقول أولًا.', report_table()
    if len(description.strip()) < 12:
        return '⚠️ الوصف قصير جدًا؛ أضف تفاصيل تساعد على المطابقة.', report_table()
    if report_type not in ('lost','found'):
        return '⚠️ اختر نوع البلاغ.', report_table()
    with db_connect() as con:
        ids=[int(r[0].split('-')[-1]) for r in con.execute('SELECT id FROM reports').fetchall()]
        rid=f'LF-{max(ids,default=0)+1:04d}'
        con.execute('INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (rid,report_type,*values,'open',datetime.now().strftime('%Y-%m-%d %H:%M')))
        con.commit()
    sync_index()
    return f'✅ تم تسجيل البلاغ **{rid}** وإضافته إلى البحث الذكي.', report_table()

def report_choices():
    return [(f"{r['id']} | {'مفقود' if r['report_type']=='lost' else 'موجود'} | {r['item']}",r['id']) for r in read_reports()]

def matches_table(report_id):
    if not report_id:
        return pd.DataFrame(columns=['رقم البلاغ','الغرض','الموقع','التشابه الرقمي','توافق الموقع','الوصف']), 'اختر بلاغًا أولًا.'
    original=next((r for r in read_reports() if r['id']==report_id),None)
    if not original:
        return pd.DataFrame(), 'البلاغ غير موجود.'
    matches=find_matches(report_id)
    df = pd.DataFrame([[r['id'],r['item'],r['location'],f"{r['score']:.3f}",'نعم' if r['location'].casefold()==original['location'].casefold() else 'لا',r['description']] for r in matches],columns=['رقم البلاغ','الغرض','الموقع','التشابه الرقمي','توافق الموقع','الوصف'])
    return df, f"🔎 ظهرت {len(matches)} مطابقة محتملة للبلاغ {report_id}. قيمة التشابه ليست احتمال ملكية أو ضمان تطابق."

def search_table(query,report_type):
    rows=semantic_search(query,report_type,7)
    return pd.DataFrame([[r['id'],r['item'],r['location'],'مفقود' if r['report_type']=='lost' else 'موجود',f"{r['score']:.3f}",r['description']] for r in rows],columns=['رقم البلاغ','الغرض','الموقع','النوع','التشابه الرقمي','الوصف'])

def change_status(report_id,new_status):
    if not report_id:
        return '⚠️ اختر بلاغًا.'
    with db_connect() as con:
        cursor=con.execute('UPDATE reports SET status = ? WHERE id = ?', (new_status,report_id))
        con.commit()
    if not cursor.rowcount:
        return '⚠️ لم يتم العثور على البلاغ.'
    sync_index()
    return f'✅ تم تحديث حالة {report_id} إلى {new_status}.'

print('✅ Dashboard and reports functions ready')

import re

def chat_reply(message, history):
    question=str(message or '').strip()
    if not question:
        return 'اكتب سؤالك عن غرض مفقود أو موجود.'
    lowered=question.casefold()
    if any(s in lowered for s in ['كم بلاغ','عدد البلاغ','how many reports','total reports']):
        rows=read_reports()
        response=f"لدينا {len(rows)} بلاغًا تجريبيًا: {sum(r['report_type']=='lost' for r in rows)} مفقودات و{sum(r['report_type']=='found' for r in rows)} موجودات."
    else:
        wants_found = any(s in lowered for s in ['found','عثرت','عُثر','موجود','وجدت','لقاها','لقوها','تم العثور','انوجد'])
        wants_lost = any(s in lowered for s in ['lost','فقدت','مفقود','ضاع','ضاعت','ضيعت'])
        report_type='found' if wants_found and not wants_lost else 'lost' if wants_lost and not wants_found else 'all'
        matches=semantic_search(question,report_type,4,only_open=True)
        if not matches:
            response='لم أجد بلاغات مناسبة ضمن السجلات التجريبية. يمكنك إضافة بلاغ جديد وتحديد وصف الغرض وموقعه.'
        else:
            top=matches[:3]
            lines=[]
            for r in top:
                name='موجود' if r['report_type']=='found' else 'مفقود'
                lines.append(f"• **{r['id']}** — {name}: **{r['item']}**، الموقع: **{r['location']}**. {r['description']} (تشابه {r['score']:.2f})")
            response='هذه أقرب البلاغات للسؤال داخل قاعدة البيانات التجريبية:\n\n'+'\n\n'.join(lines)+'\n\n⚠️ هذه نتائج محتملة. راجع تفاصيل البلاغ وتحقق من الملكية قبل التسليم.'
    with db_connect() as con:
        con.execute('INSERT INTO chat_history (question,answer,created_at) VALUES (?,?,?)',(question,response,datetime.now().isoformat(timespec='seconds')))
        con.commit()
    return response

def evaluation():
    checks=[
        ('black headphones at airport','found','LF-0002'),
        ('silver watch shopping mall','found','LF-0004'),
        ('blue backpack university library','found','LF-0006'),
        ('brown leather wallet park','found','LF-0012'),
        ('grey coat at boarding gate','found','LF-0028')
    ]
    output=[]
    ranks=[]
    for question,report_type,expected in checks:
        hits=semantic_search(question,report_type,3)
        retrieved=[r['id'] for r in hits]
        rank=retrieved.index(expected)+1 if expected in retrieved else None
        ranks.append(rank)
        output.append([question,expected,', '.join(retrieved),rank if rank else 'لم يظهر'])
    hit_rate=sum(r is not None for r in ranks)/len(ranks)
    mrr=sum(1/r if r else 0 for r in ranks)/len(ranks)
    return pd.DataFrame(output,columns=['السؤال','البلاغ المتوقع','أول 3 نتائج','ترتيب الصحيح']), f'**Hit Rate@3: {hit_rate:.0%}**   |   **MRR@3: {mrr:.2f}**  — اختبار على خمس حالات اصطناعية فقط، وليس دقة عامة للنظام.'

print('✅ Semantic chatbot and evaluation ready')

with gr.Blocks(title='Smart Lost & Found Matcher', theme=gr.themes.Soft(), css='.gradio-container {direction: rtl;} .gradio-container textarea, .gradio-container input {text-align: right;}') as demo:
    gr.Markdown('# 🧳 Smart Lost & Found Matcher\n### AI-powered Lost & Found Dashboard | لوحة بلاغات ذكية')
    gr.Markdown('بيانات تجريبية فقط — لا تُعد درجات التشابه إثباتًا للمطابقة، ولا يُسلَّم الغرض دون التحقق من ملكيته.')

    with gr.Tab('📊 Dashboard'):
        stats = gr.Markdown()
        with gr.Row():
            location_plot = gr.Plot(label='المواقع')
            type_plot = gr.Plot(label='الأنواع')
        overview_table = gr.Dataframe(label='أحدث البلاغات',interactive=False)
        refresh = gr.Button('🔄 تحديث الداشبورد')
        refresh.click(fn=overview,inputs=[],outputs=[stats,location_plot,type_plot,overview_table])

    with gr.Tab('📋 البلاغات'):
        with gr.Row():
            filter_type = gr.Dropdown(choices=[('الكل','all'),('مفقود','lost'),('موجود','found')],value='all',label='نوع البلاغ')
            filter_status = gr.Dropdown(choices=[('الكل','all'),('مفتوح','open'),('مغلق','resolved')],value='all',label='الحالة')
        filtered_reports = gr.Dataframe(value=report_table(),label='سجل البلاغات',interactive=False)
        gr.Button('بحث / تحديث القائمة').click(fn=report_table,inputs=[filter_type,filter_status],outputs=filtered_reports)

    with gr.Tab('➕ إضافة بلاغ'):
        with gr.Row():
            rtype = gr.Radio(choices=[('مفقود','lost'),('موجود','found')],value='lost',label='نوع البلاغ')
            item = gr.Textbox(label='اسم الغرض',placeholder='Black headphones')
        with gr.Row():
            location = gr.Dropdown(choices=['Airport','Mall','University','Park','Cafe','Gym','Train Station','Other'],allow_custom_value=True,label='الموقع',value='Airport')
            reporter = gr.Textbox(label='اسم مستعار لصاحب البلاغ',placeholder='Ahmed')
        description = gr.Textbox(label='وصف تفصيلي',lines=3,placeholder='Black wireless headphones in a small carrying case...')
        add_btn=gr.Button('💾 حفظ البلاغ',variant='primary')
        add_status=gr.Markdown()
        add_btn.click(fn=new_report,inputs=[rtype,item,location,description,reporter],outputs=[add_status,filtered_reports])

    with gr.Tab('🧩 المطابقة الذكية'):
        match_id=gr.Dropdown(choices=report_choices(),label='اختر بلاغًا لاقتراح البلاغات المقابلة')
        reload_choices=gr.Button('🔄 تحديث قائمة البلاغات')
        reload_choices.click(fn=lambda:gr.update(choices=report_choices()),outputs=match_id)
        match_button=gr.Button('✨ ابحث عن مطابقات محتملة',variant='primary')
        match_data=gr.Dataframe(label='البلاغات الأقرب',interactive=False)
        match_notice=gr.Markdown()
        match_button.click(fn=matches_table,inputs=match_id,outputs=[match_data,match_notice])
        gr.Markdown('⚠️ يفلتر النظام البلاغات بحسب النوع المقابل، ويستخدم تشابه الأوصاف وترجيح الموقع. هذه ليست مطابقة هوية مؤكدة.')

    with gr.Tab('🔎 البحث الدلالي'):
        search_question=gr.Textbox(label='اكتب وصفًا أو سؤالًا',value='black headphones near the airport')
        search_type=gr.Dropdown(choices=[('كل البلاغات','all'),('المفقودات فقط','lost'),('الموجودات فقط','found')],value='all',label='نطاق البحث')
        search_button=gr.Button('🔍 بحث',variant='primary')
        search_data=gr.Dataframe(label='نتائج البحث',interactive=False)
        search_button.click(fn=search_table,inputs=[search_question,search_type],outputs=search_data)

    with gr.Tab('💬 الشات بوت'):
        gr.Markdown('المساعد يبحث في بلاغاتنا التجريبية باستخدام Embeddings؛ ردوده مبنية على السجلات، وليست توليدًا حرًا بواسطة LLM.')
        gr.ChatInterface(fn=chat_reply,examples=['Where were black headphones found?','فقدت سماعات سوداء في المطار','كم بلاغ عندنا؟','Is there a lost silver watch?'])

    with gr.Tab('🧪 التقييم'):
        gr.Markdown('تقييم استرجاع بسيط باستخدام خمس أسئلة متوقعة، على غرار Lab 5.')
        eval_btn=gr.Button('📏 تشغيل تقييم الاسترجاع')
        eval_table=gr.Dataframe(interactive=False)
        eval_stats=gr.Markdown()
        eval_btn.click(fn=evaluation,outputs=[eval_table,eval_stats])

    demo.load(fn=overview,outputs=[stats,location_plot,type_plot,overview_table])

if __name__ == '__main__':
    demo.launch(share=os.environ.get('GRADIO_SHARE', 'false').lower() == 'true', debug=False, server_name='0.0.0.0')