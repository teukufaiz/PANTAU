import dash
import os
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from google import genai

# ==============================================================================
# 1. KONFIGURASI
# ==============================================================================
client = genai.Client(vertexai=True, project="project-4ad28633-a4ed-4294-94c", location="us-central1")
MODEL_ID = "gemini-2.5-flash"
FILE_PATH = 'Hasil_Analisis_COBIT_Final_normalized.xlsx'
TOTAL_COBIT_OBJECTIVES = 40

L1_NAMES = {
    '1': '1. Tata Kelola', '2': '2. Arsitektur', '3': '3. Manajemen Resiko',
    '4': '4. Ketahanan & Keamanan Siber', '5': '5. Teknologi', '6': '6. Data',
    '7': '7. Kolaborasi', '8': '8. Perlindungan Konsumen', '9': '9. Operasional TI',
    '10': '10. Jaringan Komunikasi', '11': '11. Rencana Pemulihan Bencana',
    '12': '12. Audit Intern', '13': '13. Aplication Control'
}
L1_LIST = list(L1_NAMES.values())

domain_l2_list = [
    '1.1 Tatanan Institusi', '1.2 Tata Kelola TI', '2.1 Arsitektur TI',
    '3.1 Manajemen Risiko TI', '3.2 Manajemen Risiko Operasional TI', '3.3 Manajemen Risiko Jaringan Komunikasi',
    '4.1 Tata Kelola Risiko', '4.2 Kerangka Manajemen Risiko',
    '4.3 Proses MR, Kecukupan SDM & SIMR', '4.4 Sistem Pengendalian Risiko', '4.5 Penerapan Proses Ketahanan Siber',
    '5.1 Adopsi TI', '5.2 Penggunaan Pihak Penyedia Jasa TI', '5.3 Layanan Perbankan Elektronik',
    '5.4 Layanan Digital', '5.5 Pengembangan TI', '5.6 Pengadaan TI',
    '6.1 Tata Kelola Data', '6.2 Pelindungan Data Pribadi', '6.3 Transfer Data',
    '7.1 Kerja Sama Kemitraan', '7.2 Penyedia Jasa TI oleh Bank',
    '8.1 Aspek Pelayanan', '8.2 Aspek Pelindungan',
    '9.1 Pusat Data', '9.2 Perencanaan & Pemantauan Kapasitas TI', '9.3 Pengelolaan Konfigurasi',
    '9.4 Pemeliharaan Perangkat Keras & Lunak', '9.5 Manajemen Perubahan', '9.6 Penanganan Kejadian',
    '9.7 Pengendalian Pertukaran Informasi', '9.8 Pengelolaan Library', '9.9 Disposal Perangkat Keras & Lunak',
    '10.1 Kebijakan terkait Jaringan Komunikasi',
    '11.1 Rencana Pemulihan Bencana', '11.2 Prosedur Rencana Pemulihan Bencana',
    '11.3 Pengujian Rencana Pemulihan Bencana', '11.4 Pemeliharaan Rencana Pemulihan Bencana',
    '12.1 Kebijakan IT Audit', '12.2 Proses IT Audit', '13.1 Input & Access', '13.2 File and Data Transmission', '13.3 Processing', '13.4 Output', 
    '13.5 Master File and Standing Data'
]

depts = ['ITR', 'CIS', 'DGR', 'WBA', 'IGC', 'IOP', 'ISG']
years = [2023, 2024, 2025]
risks = ['H', 'M', 'L']

COBIT_ASPECTS = {
    'EDM': '#2c3e50', 'APO': '#2980b9',
    'BAI': '#27ae60', 'DSS': '#e67e22', 'MEA': '#c0392b'
}

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
def load_data():
    try:
        raw_df = pd.read_excel(FILE_PATH)
        dff = pd.DataFrame({
            'Audit Year':       raw_df['AUDITYEAR'],
            'Quarter':          raw_df['QUARTALYEAR'].astype(str).str.upper(),
            'Departemen':       raw_df['DEPARTMENTINCHARGE_SINGKAT'],
            'Audit Assignment': raw_df['ASSIGNMENTNAME'],
            'Resiko':           raw_df['Risiko (H/M/L)'],
            'Significant':      raw_df['SIGNIFICANTFINDING (Y/N)'],
            'Finding':          raw_df['OBSERVATIONRESULT'],
            'Domain L1':        raw_df['Control Lvl 1'],
            'Domain L2':        raw_df['Control Lvl 2'],
            'COBIT_Code':       raw_df.iloc[:, 46].astype(str).str.extract(r'([A-Z]{3}\d{2})')[0],
            'COBIT_Score':      pd.to_numeric(raw_df.iloc[:, 47], errors='coerce').fillna(0),
        })
        
        return dff
    except Exception as e:
        print(f"[load_data ERROR] {e}")
        return pd.DataFrame(columns=[
            'Audit Year', 'Quarter', 'Departemen', 'Audit Assignment',
            'Resiko', 'Significant', 'Finding',
            'Domain L1', 'Domain L2', 'COBIT_Code', 'COBIT_Score'
        ])

df = load_data()

def map_to_aspect(domain_name):
    d = str(domain_name).upper()
    if any(x in d for x in ['EDM', 'TATA KELOLA']):                             return 'EDM'
    if any(x in d for x in ['APO', 'PERENCANAAN', 'ARSITEKTUR', 'RESIKO']):    return 'APO'
    if any(x in d for x in ['BAI', 'PENGEMBANGAN', 'TEKNOLOGI']):              return 'BAI'
    if any(x in d for x in ['DSS', 'OPERASIONAL', 'JARINGAN', 'SIBER', 'PEMULIHAN']): return 'DSS'
    if any(x in d for x in ['MEA', 'AUDIT']):                                  return 'MEA'
    return 'DSS'

def apply_filters(year_idx, dept_val, risk_vals, sig_val, mode):
    dff = df.copy()
    y_opts = ['All'] + years
    if y_opts[year_idx] != 'All':
        dff = dff[dff['Audit Year'] == y_opts[year_idx]]
    if dept_val != 'All':
        dff = dff[dff['Departemen'] == dept_val]
    if risk_vals:
        dff = dff[dff['Resiko'].isin(risk_vals)]
    if mode == 'significant':
        dff = dff[dff['Significant'] == 'Y']
    elif sig_val != 'All':
        dff = dff[dff['Significant'] == sig_val]
    return dff

# ==============================================================================
# 3. LAYOUT FUNCTIONS
# ==============================================================================
def login_layout():
    return html.Div([
        # LEFT PANEL
        html.Div([
            html.Div(className="grid-bg"),
            html.Div(className="glow-orb orb-1"),
            html.Div(className="glow-orb orb-2"),
            html.Div(className="glow-orb orb-3"),
            html.Div([
                html.Div([
                    html.Div("🔥", className="brand-icon"),
                    html.Div("PANTAU", className="brand-name")
                ], className="brand-logo"),
            ], className="brand"),
            html.Div([
                html.H1(["Platform Analitik", html.Br(), html.Span("Temuan Audit.")]),
                html.P("Visualisasi temuan audit IT secara real-time, analisis maturity COBIT 2019, dan insight strategis berbasis Vertex AI — dalam satu platform terpadu.")
            ], className="hero-text"),
            html.Div([
                html.Span("COBIT 2019", className="tag"),
                html.Span("Heatmap Analysis", className="tag"),
                html.Span("Maturity Index", className="tag"),
                html.Span("Vertex AI", className="tag"),
            ], className="floating-tags"),
            html.Div([
                html.Div([html.Div("1,714", className="stat-num"), html.Div("Total Temuan", className="stat-label")], className="stat-item"),
                html.Div(className="stat-divider"),
                html.Div([html.Div("3", className="stat-num"), html.Div("Tahun Audit", className="stat-label")], className="stat-item"),
                html.Div(className="stat-divider"),
                html.Div([html.Div("40", className="stat-num"), html.Div("COBIT Objectives", className="stat-label")], className="stat-item"),
            ], className="stats-row"),
        ], className="left-panel"),

        # RIGHT PANEL
        html.Div([
            html.Div(className="scanline"),
            html.Div([
                html.H2("Selamat Datang"),
                html.P("Masuk dengan kredensial yang diberikan oleh administrator.")
            ], className="login-header"),
            html.Div([
                html.Label("Username", className="form-label"),
                html.Div([
                    dbc.Input(id="username", placeholder="Masukkan username", type="text", className="form-input"),
                    html.Span("👤", className="input-icon")
                ], className="input-wrap")
            ], className="form-group"),
            html.Div([
                html.Label("Password", className="form-label"),
                html.Div([
                    dbc.Input(id="password", placeholder="Masukkan password", type="password", className="form-input"),
                    html.Span("🔑", className="input-icon")
                ], className="input-wrap")
            ], className="form-group"),
            dbc.Button("Masuk ke Platform →", id="login-button", className="btn-login"),
            html.Div(id="login-alert", className="mt-2"),
            html.Div([html.Span("Keamanan Sistem")], className="divider"),
            html.Div([html.Span("🔒"), html.Span("Koneksi terenkripsi · Sesi otomatis berakhir · Akses terbatas")], className="security-badge"),
            html.Div(["© 2025 Internal Audit IT Division", html.Br(), "Akses tidak sah akan dicatat dan dilaporkan."], className="footer-note"),
        ], className="right-panel"),
    ], className="login-wrapper")

def dashboard_layout():
    # Helper komponen untuk filter 
    FILTER_CARD = dbc.Card([
        dbc.CardHeader([
            html.Span("⚙️ ", className="me-1"),
            html.Strong("Filter Granular")
        ]),
        dbc.CardBody([
            html.Label("Domain Level:", className="small fw-bold"),
            dcc.Slider(id="lvl-slider", min=1, max=2, step=1, value=1,
                       marks={1: 'L1', 2: 'L2'}),
            html.Hr(className="my-2"),

            html.Label("Tahun Audit:", className="small fw-bold"),
            dcc.Slider(id="year-slider", min=0, max=len(years), step=1,
                       marks={i: str(y) for i, y in enumerate(['All'] + years)}, value=0),
            html.Br(),

            html.Label("Departemen:", className="small fw-bold"),
            dcc.Dropdown(
                id="dept-dropdown",
                options=[{'label': 'All', 'value': 'All'}] + [{'label': d, 'value': d} for d in depts],
                value='All', clearable=False
            ),
            html.Br(),

            html.Label("Tingkat Resiko:", className="small fw-bold"),
            dbc.Checklist(
                id="risk-check",
                options=[{'label': r, 'value': r} for r in risks],
                value=risks, inline=True
            ),

            html.Div([
                html.Hr(className="my-2"),
                html.Label("Temuan Signifikan:", className="small fw-bold"),
                dbc.RadioItems(
                    id="sig-toggle",
                    options=[{'label': 'All', 'value': 'All'},
                             {'label': 'Y', 'value': 'Y'},
                             {'label': 'N', 'value': 'N'}],
                    value='All', inline=True
                ),
            ], id="sig-filter-container"),

            html.Hr(className="my-3"),
            dbc.Button(
                [html.I(className="me-2"), "🚀 Analyze Findings"],
                id="analyze-btn", color="primary",
                className="w-100 fw-bold", n_clicks=0
            ),
            html.Small(
                "Klik untuk generate analisis AI berdasarkan filter saat ini.",
                className="text-muted d-block text-center mt-1"
            )
        ])
    ], className="shadow-sm sticky-top", style={"top": "1rem"})

    return dbc.Container([
        # ── NAVBAR ──────────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("🔥 Platform Analitik Temuan Audit", className="mb-0 text-white"),
                    html.Small("COBIT 2019 · Maturity & Heatmap Analysis · Vertex AI", className="text-white-50")
                ], className="py-3 px-3", style={"background": "linear-gradient(135deg,#1a3c5e,#2e86ab)"})
            ], width=12)
        ], className="mb-3"),

        # ── MODE TOGGLE ──────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Label("Visualisasi Berdasarkan:", className="fw-bold me-3"),
                        dbc.RadioItems(
                            id="mode-toggle",
                            options=[
                                {"label": "🔢 Banyak Temuan",      "value": "count"},
                                {"label": "⚠️ Significant Finding", "value": "significant"}
                            ],
                            value="count", inline=True, className="d-inline-block"
                        )
                    ], className="py-2")
                ], className="mb-3 shadow-sm")
            ], width=12)
        ]),

        # ── MAIN BODY ────────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id="heatmap-graph", config={'displayModeBar': False})
                    ])
                ], className="shadow-sm")
            ], width=9),
            dbc.Col([FILTER_CARD], width=3)
        ]),

        # ── DETAIL MODAL ─────────────────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
            dbc.ModalBody([
                dbc.InputGroup([
                    dbc.InputGroupText("🔍"),
                    dbc.Input(id="search-input", placeholder="Cari Assignment...", type="text"),
                ], className="mb-4 shadow-sm"),
                html.Div(id="modal-content-accordion")
            ]),
            dbc.ModalFooter(dbc.Button("Close", id="close-modal", n_clicks=0))
        ], id="detail-modal", size="lg", is_open=False, scrollable=True),

        html.Footer([
            html.Hr(),
            html.Div([
                html.Span("🤖 Powered by ", className="me-1"),
                
                html.Img(
                    src="/assets/vertexai.png",
                    style={
                        "height": "90px",
                        "verticalAlign": "middle"
                    }
                ),
            ], className="text-center text-muted medium py-2")
        ]),

        # ── ANALYSIS SECTION ─────────────────────────────────────────────────────
        html.Div(id="analysis-section", children=[
            html.Hr(className="my-4"),
            html.H4("📊 Hasil Analisis Strategis", className="text-primary mb-1"),
            html.Small(id="analysis-context-label", className="text-muted d-block mb-4"),

            dbc.Row(id="kpi-row", className="mb-4 g-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📊 Maturity per Aspek COBIT", className="text-primary fw-bold"),
                            dcc.Graph(id="aspects-bar", config={'displayModeBar': False}, style={'height': '400px'})
                        ])
                    ], className="shadow-sm")
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("🎯 COBIT 2019 Audit Coverage", className="text-primary fw-bold"),
                            dcc.Graph(id="coverage-gauge", config={'displayModeBar': False}, style={'height': '300px'}),
                        ])
                    ], className="shadow-sm h-100")
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📈 Overall Maturity Index", className="text-success fw-bold"),
                            dcc.Graph(id="maturity-gauge", config={'displayModeBar': False}, style={'height': '300px'}),
                        ])
                    ], className="shadow-sm h-100")
                ], width=6),
            ], className="mb-4 g-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("🔍 Distribusi Temuan per COBIT Domain", className="text-primary fw-bold"),
                            dcc.Graph(id="sunburst-chart", config={'displayModeBar': False}, style={'height': '550px'})
                        ])
                    ], className="shadow-sm")
                ], width=12)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("🚫 L1 — Domain Belum Tercakup", className="text-danger fw-bold"),
                            html.Div(id="gap-l1-badges", style={"minHeight": "100px"})
                        ])
                    ], className="shadow-sm h-100")
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("⚠️ L2 — Kontrol Belum Pernah Dicek", className="text-warning fw-bold"),
                            html.Div(id="gap-l2-list", style={"maxHeight": "200px", "overflowY": "auto"})
                        ])
                    ], className="shadow-sm h-100")
                ], width=6),
            ], className="mb-4 g-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("🤖 ", className="me-1"),
                            html.Strong("Analisis Strategis — Vertex AI")
                        ], className="bg-primary text-white"),
                        dbc.CardBody([
                            dcc.Loading(
                                html.Div(id="ai-output", className="p-2", style={"minHeight": "150px"}),
                                type="circle", color="#2e86ab"
                            )
                        ])
                    ], className="shadow")
                ], width=12)
            ], className="mb-5"),

        ], style={"display": "none"}),
        dcc.Store(id="filter-snapshot"),
        
    ], fluid=True, style={'backgroundColor': '#f8f9fa'})

# ==============================================================================
# 4. DASH INITIALIZATION (CRITICAL FIX FOR ERROR)
# ==============================================================================
# Tambahkan suppress_callback_exceptions=True agar tidak error saat login
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)
app.title = "Platform Analitik Temuan Audit"

app.layout = html.Div([
    dcc.Store(id='login-status', storage_type='session', data={'logged_in': False}),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# ==============================================================================
# 5. CALLBACKS (AUTH & NAVIGATION)
# ==============================================================================

@app.callback(
    Output('page-content', 'children'),
    Input('login-status', 'data')
)
def render_page(auth_data):
    if auth_data and auth_data.get('logged_in'):
        return dashboard_layout()
    return login_layout()

@app.callback(
    [Output('login-status', 'data'), Output('login-alert', 'children')],
    Input('login-button', 'n_clicks'),
    [State('username', 'value'), State('password', 'value')],
    prevent_initial_call=True
)
def handle_auth(n, user, pw):
    if n is None or n == 0:
        return dash.no_update, ""

    # 2. Logika validasi
    if user == "admin" and pw == "admin123":
        return {'logged_in': True}, ""
    
    # 3. Jika salah (hanya muncul setelah diklik)
    return {'logged_in': False}, dbc.Alert("Username atau Password salah!", color="danger", className="mt-2")

# ==============================================================================
# 6. DASHBOARD CALLBACKS (LOGIKA ASLI 700+ BARIS)
# ==============================================================================

@app.callback(
    Output("sig-filter-container", "style"),
    Input("mode-toggle", "value")
)
def toggle_sig_filter(mode):
    return {"display": "none"} if mode == "significant" else {"display": "block"}

@app.callback(
    Output("heatmap-graph", "figure"),
    [Input("mode-toggle", "value"), Input("lvl-slider", "value"),
     Input("year-slider", "value"), Input("dept-dropdown", "value"),
     Input("risk-check", "value"), Input("sig-toggle", "value")]
)
def update_heatmap(mode, lvl, year_idx, dept_val, risk_vals, sig_val):
    dff = apply_filters(year_idx, dept_val, risk_vals, sig_val, mode)

    y_col = 'Domain L1' if lvl == 1 else 'Domain L2'
    full_y_list = L1_LIST if lvl == 1 else domain_l2_list
    
    pivot = dff.groupby([y_col, 'Quarter']).size().unstack(fill_value=0)
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        if q not in pivot.columns: pivot[q] = 0
    pivot = pivot[['Q1', 'Q2', 'Q3', 'Q4']].reindex(full_y_list, fill_value=0)

    mask_13 = pivot.index.str.startswith('13.')
    pivot.loc[mask_13] = (pivot.loc[mask_13] / 2.5).round().astype(int)
    
    z_data = pivot.values.astype(float)
    z_display = np.where(z_data == 0, np.nan, z_data)  # hanya untuk warna, bukan teks
    

    fig = go.Figure(data=go.Heatmap(
        z=z_display,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, 'rgb(255,255,204)'], [0.5, 'rgb(253,141,60)'], [1, 'rgb(189,0,38)']],
        zmin=1,                          # ← kunci skala mulai dari 1
        zmax=pivot.values.max(),         # ← kunci skala maksimum dari data aktual
        text=z_data,                     # ← teks tetap pakai angka asli (bukan NaN)
        texttemplate="%{text:.0f}",      # ← format integer, bukan float
        hovertemplate="Domain: %{y}<br>Kuartal: %{x}<br>Jumlah: %{customdata}<extra></extra>",
        customdata=z_data,               # ← hover juga pakai angka asli
        showscale=True,
        connectgaps=False
    ))
    fig.update_layout(
        height=800 if lvl == 2 else 580, margin=dict(l=260, t=20, r=20, b=40),
        xaxis_title="Kuartal Audit", yaxis_title=y_col,
        paper_bgcolor='white', plot_bgcolor='white'
    )
    return fig

@app.callback(
    [Output("detail-modal", "is_open"),
     Output("modal-title", "children"),
     Output("modal-content-accordion", "children")],
    [Input("heatmap-graph", "clickData"),
     Input("close-modal", "n_clicks"),
     Input("search-input", "value")],
    [State("detail-modal", "is_open"), State("lvl-slider", "value"),
     State("year-slider", "value"), State("dept-dropdown", "value"),
     State("risk-check", "value"), State("sig-toggle", "value"),
     State("mode-toggle", "value")]
)
def handle_modal(clickData, n_close, search_val, is_open, lvl, y_idx, dept_val, risk_vals, sig_val, mode):
    ctx = callback_context
    if (not ctx.triggered or not clickData or ctx.triggered[0]["prop_id"].split(".")[0] == "close-modal"):
        return False, "", ""
    y_val = clickData["points"][0]["y"]
    x_val = clickData["points"][0]["x"]
    y_col = 'Domain L1' if lvl == 1 else 'Domain L2'
    details = apply_filters(y_idx, dept_val, risk_vals, sig_val, mode)
    details = details[(details[y_col] == y_val) & (details['Quarter'] == x_val)]
    if search_val:
        details = details[details['Audit Assignment'].str.contains(search_val, case=False, na=False)]
    accordion_items = [
        dbc.AccordionItem([
            html.Div([
                html.P([html.B("Dept: "), row['Departemen'], html.B(" | Risk: "), row['Resiko'], html.B(" | Sig: "), row['Significant']]),
                html.P(row['Finding'], className="text-muted small")
            ]) for _, row in details[details['Audit Assignment'] == assign].iterrows()
        ], title=assign) for assign in details['Audit Assignment'].unique()
    ]
    return True, f"📋 Rincian: {y_val} ({x_val})", dbc.Accordion(accordion_items, flush=True)

@app.callback(
    Output("filter-snapshot", "data"),
    Input("analyze-btn", "n_clicks"),
    [State("year-slider", "value"), State("dept-dropdown", "value"),
     State("risk-check", "value"), State("sig-toggle", "value"),
     State("mode-toggle", "value"), State("lvl-slider", "value")],
    prevent_initial_call=True
)
def snapshot_filters(n, year_idx, dept_val, risk_vals, sig_val, mode, lvl):
    return {"year_idx": year_idx, "dept_val": dept_val, "risk_vals": risk_vals, "sig_val": sig_val, "mode": mode, "lvl": lvl}

@app.callback(
    [Output("analysis-section", "style"),
     Output("analysis-context-label", "children"),
     Output("kpi-row", "children"),
     Output("aspects-bar", "figure"),
     Output("coverage-gauge", "figure"),
     Output("maturity-gauge", "figure"),
     Output("sunburst-chart", "figure"),
     Output("gap-l1-badges", "children"),
     Output("gap-l2-list", "children")],
    Input("filter-snapshot", "data"),
    prevent_initial_call=True
)
def update_analysis_charts(snap):
    if not snap: return {"display": "none"}, "", [], go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], []
    dff = apply_filters(snap["year_idx"], snap["dept_val"], snap["risk_vals"], snap["sig_val"], snap["mode"])
    dff['Aspect'] = dff['Domain L1'].apply(map_to_aspect)
    y_opts = ['All'] + years
    ctx_label = f"Filter aktif → Tahun: {y_opts[snap['year_idx']]} | Dept: {snap['dept_val']} | Total: {len(dff)}"
    
    total_findings, high_risk_count, sig_count = len(dff), len(dff[dff['Resiko'] == 'H']), len(dff[dff['Significant'] == 'Y'])
    avg_maturity = round(5 - dff['COBIT_Score'].mean(), 2) if not dff.empty else 0
    coverage_count = dff['COBIT_Code'].nunique()

    kpi_data = [("🔢 Total Temuan", str(total_findings), "primary"), ("🔴 High Risk", str(high_risk_count), "danger"), ("⚠️ Signifikan (Y)", str(sig_count), "warning"), ("📈 Maturity Index", f"{avg_maturity}/5.0", "success"), ("🎯 COBIT Coverage", f"{coverage_count}/40", "info")]
    kpi_cards = [dbc.Col(dbc.Card([dbc.CardBody([html.H3(val, className=f"text-{color} mb-0 fw-bold"), html.P(label, className="text-muted small mb-0")], className="text-center py-2")], className="shadow-sm"), width=True) for label, val, color in kpi_data]

    aspect_scores = dff.groupby('Aspect')['COBIT_Score'].mean().reset_index()
    aspect_scores['Maturity'] = (5 - aspect_scores['COBIT_Score']).round(2)
    fig_bar = px.bar(aspect_scores, x='Aspect', y='Maturity', color='Aspect', text='Maturity', color_discrete_map=COBIT_ASPECTS)
    fig_bar.update_layout(yaxis=dict(range=[0, 5.5]), showlegend=False, height=320, plot_bgcolor='white')

    fig_cov = go.Figure(go.Indicator(mode="gauge+number", value=coverage_count, gauge={'axis': {'range': [0, 40]}}))
    fig_mat = go.Figure(go.Indicator(mode="gauge+number", value=avg_maturity, gauge={'axis': {'range': [0, 5]}}))
    fig_sun = px.sunburst(dff[dff['COBIT_Score'] > 0], path=['Aspect', 'Domain L1'], values='COBIT_Score', color='Aspect', color_discrete_map=COBIT_ASPECTS)

    checked_l1 = set(dff['Domain L1'].astype(str).str.upper())
    gap_l1 = [d for d in L1_LIST if not any(d.upper() in c or c in d.upper() for c in checked_l1)]
    l1_badges = [dbc.Badge(d, color="danger", className="me-1 mb-1 p-2") for d in gap_l1] or [html.Span("✅ Lengkap", className="text-success")]
    
    gap_l2 = [d for d in domain_l2_list if d not in set(dff['Domain L2'].astype(str))]
    l2_list = html.Ul([html.Li(d, className="small text-muted") for d in gap_l2])

    return {"display": "block"}, ctx_label, kpi_cards, fig_bar, fig_cov, fig_mat, fig_sun, l1_badges, l2_list

@app.callback(
    Output("ai-output", "children"),
    Input("filter-snapshot", "data"),
    prevent_initial_call=True
)
def run_ai_analysis(snap):
    if not snap: return ""
    dff = apply_filters(snap["year_idx"], snap["dept_val"], snap["risk_vals"], snap["sig_val"], snap["mode"])
    
    # Prompt Lengkap
    prompt = f"Lakukan analisis audit IT mendalam untuk data: Total Temuan {len(dff)}, High Risk {len(dff[dff['Resiko']=='H'])}, Avg Maturity {round(5-dff['COBIT_Score'].mean(),2)}. Berikan rekomendasi strategis."
    
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return dcc.Markdown(response.text, className="p-2")
    except Exception as e:
        return dbc.Alert(f"❌ Error Vertex AI: {str(e)}", color="danger")

# ==============================================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)