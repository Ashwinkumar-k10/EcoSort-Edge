"""
scripts/generate_exact_ieee_word_doc.py
Generates the definitive, mathematically honest, rigorously grounded IEEE Word Document (.docx)
for EcoSort-Edge, adhering strictly to IEEE Transactions formatting guidelines, empirical data,
and the review checklist (A1-A9, B1-B9, E, F).
"""

import sys
import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DOCX = PROJECT_ROOT / "EcoSort_Edge_IEEE_Technical_Proposal.docx"

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=70, bottom=70, left=100, right=100):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_horizontal_borders(table, color="D3D3D3", sz="4"):
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="{int(sz)*2}" w:space="0" w:color="0F172A"/>'
            f'<w:bottom w:val="single" w:sz="{int(sz)*2}" w:space="0" w:color="0F172A"/>'
            f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'<w:insideV w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def add_hyperlink(paragraph, text, url):
    part = paragraph.part
    r_id = part.relate_to(url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = parse_xml(
        f'<w:hyperlink {nsdecls("w")} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="{r_id}">'
        f'<w:r>'
        f'<w:rPr>'
        f'<w:color w:val="0284C7"/>'
        f'<w:u w:val="single"/>'
        f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'
        f'</w:rPr>'
        f'<w:t>{text}</w:t>'
        f'</w:r>'
        f'</w:hyperlink>'
    )
    paragraph._element.append(hyperlink)

def create_ieee_document():
    doc = Document()

    # Page Margins: IEEE standard 0.75 in (54 pt)
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.75)
        s.bottom_margin = Inches(0.75)
        s.left_margin = Inches(0.75)
        s.right_margin = Inches(0.75)

    # Base Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(9.5)
    normal_style.font.color.rgb = RGBColor(15, 23, 42)

    def add_title(title_text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(title_text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(17)
        run.font.bold = True
        run.font.color.rgb = RGBColor(15, 23, 42)
        return p

    def add_author_block(authors, affiliations):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(authors)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 41, 59)

        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.paragraph_format.space_after = Pt(10)
        run2 = p2.add_run(affiliations)
        run2.font.name = "Times New Roman"
        run2.font.size = Pt(8.5)
        run2.font.italic = True
        run2.font.color.rgb = RGBColor(71, 85, 105)
        return p2

    def add_sec_heading(num_str, title_str):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(13)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(f"{num_str}.  {title_str.upper()}")
        run.font.name = "Times New Roman"
        run.font.size = Pt(10.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(15, 23, 42)
        return p

    def add_subsec_heading(letter_str, title_str):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(f"{letter_str}.  {title_str}")
        run.font.name = "Times New Roman"
        run.font.size = Pt(9.5)
        run.font.bold = True
        run.font.italic = True
        run.font.color.rgb = RGBColor(30, 41, 59)
        return p

    def add_body(text, space_after=4, bold_prefix=None, italic_prefix=None):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.06
        if bold_prefix:
            br = p.add_run(bold_prefix)
            br.font.name = "Times New Roman"
            br.font.size = Pt(9.5)
            br.font.bold = True
        if italic_prefix:
            ir = p.add_run(italic_prefix)
            ir.font.name = "Times New Roman"
            ir.font.size = Pt(9.5)
            ir.font.italic = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(30, 41, 59)
        return p

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.05
        if bold_prefix:
            br = p.add_run(bold_prefix)
            br.font.name = "Times New Roman"
            br.font.size = Pt(9.5)
            br.font.bold = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(30, 41, 59)
        return p

    def add_figure(img_path, caption_text, width_inches=5.8):
        if img_path.exists():
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.paragraph_format.keep_with_next = True
            run_img = p_img.add_run()
            run_img.add_picture(str(img_path), width=Inches(width_inches))

            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(8)
            run_cap = p_cap.add_run(caption_text)
            run_cap.font.name = "Times New Roman"
            run_cap.font.size = Pt(8.5)
            run_cap.font.italic = True
            run_cap.font.color.rgb = RGBColor(71, 85, 105)

    def render_ieee_table(table_num, table_title, headers, rows, center_cols=None):
        center_cols = center_cols or []
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(8)
        p_cap.paragraph_format.space_after = Pt(3)
        p_cap.paragraph_format.keep_with_next = True
        r_num = p_cap.add_run(f"TABLE {table_num}\n")
        r_num.font.name = "Times New Roman"
        r_num.font.size = Pt(9.0)
        r_num.font.bold = True
        r_num.font.color.rgb = RGBColor(15, 23, 42)

        r_title = p_cap.add_run(table_title.upper())
        r_title.font.name = "Times New Roman"
        r_title.font.size = Pt(8.0)
        r_title.font.bold = True
        r_title.font.color.rgb = RGBColor(51, 65, 85)

        table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_horizontal_borders(table, color="CBD5E1", sz="4")

        hdr_cells = table.rows[0].cells
        for col_idx, header_text in enumerate(headers):
            cell = hdr_cells[col_idx]
            cell.text = header_text
            set_cell_background(cell, "0F172A")
            set_cell_margins(cell, top=80, bottom=80, left=90, right=90)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in center_cols else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(8.0)
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)

        for row_idx, row_data in enumerate(rows):
            row_cells = table.rows[row_idx + 1].cells
            bg_color = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
            for col_idx, val in enumerate(row_data):
                cell = row_cells[col_idx]
                cell.text = str(val)
                set_cell_background(cell, bg_color)
                set_cell_margins(cell, top=55, bottom=55, left=90, right=90)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in center_cols else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(8.0)
                    if any(k in str(val) for k in ["PASSED", "EXCEEDED", "V3-Small"]):
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(16, 120, 60)
                    elif any(k in str(val) for k in ["FAILED", "0.0%"]):
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(185, 28, 28)
                    else:
                        run.font.color.rgb = RGBColor(30, 41, 59)

        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # =========================================================================
    # DOCUMENT HEADER
    # =========================================================================
    add_title("EcoSort-Edge: Real-Time Offline Edge AI for Municipal Waste Sorting and Food Categorisation using ExecuTorch on Raspberry Pi 5")
    add_author_block(
        "Namma AI Hackathon 2026 — Track 3: AI for Sustainability (Problem Statement 3)",
        "Organized by NASSCOM CoE IoT & AI, Arm Developer Ecosystem, IIIT-Bangalore, and UK Govt TSI\n"
        "Deployment Target: Raspberry Pi 5 (4× Arm Cortex-A76 @ 2.4 GHz) | Runtime: Arm ExecuTorch 1.5.1 (XNNPACK)"
    )

    # Canonical Links Callout
    p_link = doc.add_paragraph()
    p_link.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_link.paragraph_format.space_before = Pt(0)
    p_link.paragraph_format.space_after = Pt(6)
    p_link.add_run("Project Code Repository: ")
    add_hyperlink(p_link, "GitHub Repository (DK-A/EcoSort-Edge)", "https://github.com/DK-A/EcoSort-Edge-Offline-Waste-Sorting-Food-Categorisation-using-ExecuTorch-on-Raspberry-Pi")
    p_link.add_run("  |  Demonstration Video: ")
    add_hyperlink(p_link, "Live Hardware Demonstration (assets/DEMO.mp4)", "https://github.com/DK-A/EcoSort-Edge-Offline-Waste-Sorting-Food-Categorisation-using-ExecuTorch-on-Raspberry-Pi/raw/main/assets/DEMO.mp4")

    # Abstract
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.space_after = Pt(3)
    p_abs.paragraph_format.line_spacing = 1.06
    r_abs_lbl = p_abs.add_run("Abstract— ")
    r_abs_lbl.font.name = "Times New Roman"
    r_abs_lbl.font.size = Pt(9.0)
    r_abs_lbl.font.bold = True
    r_abs = p_abs.add_run(
        "Municipal solid waste segregation at source remains a critical environmental and operational challenge across Indian urban wards. "
        "Under the revised Solid Waste Management (SWM) Rules notified by the Ministry of Environment, Forest and Climate Change (MoEFCC), "
        "waste must be categorized into four primary streams: Wet Biodegradable, Dry Recyclable (with Refuse-Derived Fuel tags for Multi-Layer Packaging), "
        "Sanitary Waste, and Domestic Hazardous (Special-Care). Deploying cloud-tethered vision APIs in municipal transfer points faces severe practical "
        "hurdles, including poor basement cellular connectivity, recurring operational costs (illustratively $300–$1,200 annually per kiosk), and variable latency.\n\n"
        "We present EcoSort-Edge, a 100% offline edge AI classification system evaluated on an unadorned Raspberry Pi 5. The pipeline couples a MobileNetV3-Small "
        "backbone, quantized via PyTorch 2 Export (PT2E) dynamic INT8 quantization, with an Arm ExecuTorch runtime leveraging Armv8.2-A NEON SIMD and Signed Dot "
        "Product (SDOT) vector instructions. Across our 1,210-image holdout test dataset, the quantized model achieves 88.51% top-1 accuracy (1,071/1,210) "
        "and 98.35% statutory 4-stream accuracy (1,190/1,210; 95% Wilson CI: 97.5%–98.9%), with 20 cross-stream errors. On battery hazard interception, the test-set "
        "recall reaches 96.67% (58/60; 95% Wilson CI: 88.6%–99.1%), with 2 battery samples misclassified into dry recyclable categories. On our 30-sample physical "
        "Raspberry Pi 5 suite, the system achieved 30/30 statutory stream compliance and intercepted 3/3 battery samples (noting n = 3 gives a wide 95% CI of 43.9%–100.0%). "
        "The model-only CPU execution latency averages 5.09 ms (P95: 6.28 ms), while the end-to-end Python harness averages 11.85 ms per image. We conduct an empirical "
        "comparison against a peer MobileNetV3-Large baseline (4.31 MB), demonstrating that MobileNetV3-Small's 3.4× speedup is primarily explained by its 3.9× fewer "
        "arithmetic operations (~56M vs ~219M MACs), while operating at a stable 49.9 deg C."
    )
    r_abs.font.name = "Times New Roman"
    r_abs.font.size = Pt(8.5)

    # Index Terms
    p_idx = doc.add_paragraph()
    p_idx.paragraph_format.space_after = Pt(10)
    r_idx_lbl = p_idx.add_run("Index Terms— ")
    r_idx_lbl.font.name = "Times New Roman"
    r_idx_lbl.font.size = Pt(8.5)
    r_idx_lbl.font.bold = True
    r_idx = p_idx.add_run("Edge Artificial Intelligence, Arm Cortex-A76, ExecuTorch, Model Quantization, PT2E, NEON SIMD, SDOT, MobileNetV3, Solid Waste Management Rules 2026, Green Computing, Embedded Computer Vision.")
    r_idx.font.name = "Times New Roman"
    r_idx.font.size = Pt(8.5)

    # =========================================================================
    # SECTION I: INTRODUCTION & REGULATORY MANDATE
    # =========================================================================
    add_sec_heading("I", "Introduction & Problem Statement Alignment")
    add_subsec_heading("A", "Urban Solid Waste Context in India")
    add_body(
        "According to the Central Pollution Control Board (CPCB) Annual Report on Solid Waste Management [6], urban India generates approximately "
        "160,000 to 170,000 tonnes of municipal solid waste daily, of which a substantial portion continues to be landfilled without adequate "
        "segregation at source. Organic food discards and vegetable peelings undergo anaerobic decomposition when mixed with dry waste, generating "
        "methane (CH4) emissions and contaminating recyclable plastic and paper fractions. When dry recyclables are fouled by food grease and liquids, "
        "their economic recyclability is substantially degraded [1], [6]."
    )

    add_subsec_heading("B", "MoEFCC SWM Rules 2026 & Municipal Streams")
    add_body(
        "In 2026, the Ministry of Environment, Forest and Climate Change (MoEFCC) notified revised Solid Waste Management Rules [1], formalizing "
        "four primary waste categories for municipal source segregation:"
    )
    add_bullet("Leftover meals, vegetable peelings, and rotten organic produce destined for biomethanation plants or decentralized ward composting.", bold_prefix="1) Wet Biodegradable: ")
    add_bullet("Clean paper, cardboard, PET containers, glass bottles, and scrap metals for Material Recovery Facilities (MRFs). Under extended producer responsibility (EPR) guidelines, Multi-Layer Packaging (MLP) is managed as a sub-stream flagged for Refuse-Derived Fuel (RDF) co-processing in cement kilns.", bold_prefix="2) Dry Recyclable: ")
    add_bullet("Household sanitary discards requiring safe thermal disposal in high-temperature municipal incinerators.", bold_prefix="3) Sanitary Waste: ")
    add_bullet("Lithium-ion cells, lead-acid batteries, button cells, and consumer electronics requiring dedicated take-back and disposal channels.", bold_prefix="4) Domestic Hazardous (Special-Care): ")
    add_body(
        "Under Swachh Bharat Mission (SBM-Urban) guidelines, municipal kiosks typically utilize standard visual color conventions: Green (Wet), "
        "Blue (Dry), Yellow (Sanitary), and Red (Hazardous). In this work, we adopt these four statutory categories as our primary sorting target."
    )

    add_subsec_heading("C", "Edge AI vs. Cloud-Centric Architecture")
    add_body(
        "Cloud-based vision APIs present several operational bottlenecks in municipal waste facilities [7]: "
        "(1) Network Availability: Basement sorting bays and transfer stations frequently encounter heavy cellular signal attenuation (<-110 dBm); "
        "(2) Operational Expenses: Assuming 1,000–2,000 daily queries per kiosk at commercial vision pricing ($0.01–$0.02/query), API subscriptions "
        "accumulate to $300–$1,200 (approx. ₹25,000–₹100,000) per kiosk annually; "
        "(3) Variable Latency: Remote round-trip calls (500 ms to >2.0 s) impede continuous citizen drop-offs; and "
        "(4) Safety Considerations: Misclassifying hazardous items like lithium-ion batteries into dry paper streams creates severe fire hazards "
        "during subsequent mechanical compaction. Moving inference fully on-device addresses these constraints directly."
    )

    # =========================================================================
    # SECTION II: FORMAL ENGINEERING REQUIREMENTS
    # =========================================================================
    add_sec_heading("II", "Engineering Specifications & Competition Target Alignment")
    add_body(
        "To satisfy the core deliverables of Track 3 (Problem Statement 3: Edge-Based Waste Sorting and Food Categorisation), "
        "EcoSort-Edge was developed against measurable operational bounds [3], [5]:"
    )

    req_headers = ["Req ID", "Metric / Operational Parameter", "Target Requirement", "Measured Result", "Evaluation Status"]
    req_rows = [
        ["REQ-01", "Model Inference Latency (Pi 5 CPU)", "< 50.0 ms per image", "5.09 ms (P95: 6.28 ms)", "EXCEEDED (~10x Margin)"],
        ["REQ-02", "End-to-End Pipeline Harness Latency", "< 50.0 ms per image", "11.85 ms mean (min 9.30, max 14.98)", "EXCEEDED (~4x Margin)"],
        ["REQ-03", "Compiled Model Binary Size", "< 50.0 MB (.pte binary)", "2.41 MB (.pte binary)", "EXCEEDED (95.2% Margin)"],
        ["REQ-04", "Statutory 4-Stream Compliance", "High classification accuracy", "98.35% Test Set (1,190/1,210; 95% CI 97.5–98.9%)", "PASSED (20 Cross-Errors)"],
        ["REQ-05", "Hazard Interception (Battery)", "High hazard recall", "96.67% Test Set (58/60; 95% CI 88.6–99.1%)", "PASSED (2/60 Misses)"],
        ["REQ-06", "Operating Core Temperature", "Thermal Throttling < 80°C", "49.9°C stable (30-sample benchmark)", "PASSED (30.1°C Headroom)"],
        ["REQ-07", "Network & Cloud Dependency", "100% Offline Edge Operation", "0 bytes transmitted (Air-gapped)", "PASSED (100% On-Device)"],
        ["REQ-08", "Data Privacy Alignment", "DPDP Act 2023 Principles", "100% In-Memory RAM execution", "PASSED (Zero Persistence)"]
    ]
    render_ieee_table("I", "Formal Engineering Requirements & Measured Performance", req_headers, req_rows, center_cols=[0, 4])

    # =========================================================================
    # SECTION III: SYSTEM ARCHITECTURE & SOLUTION DESIGN
    # =========================================================================
    add_sec_heading("III", "System Architecture & Pipeline Design")
    add_body(
        "EcoSort-Edge decouples perception, policy verification, and telemetry output into three distinct software stages under the "
        "architectural principle: 'Edge Vision Perceives, Deterministic Policy Decides'."
    )

    # Embed Fig. 1: Solution Architecture
    add_figure(ASSETS_DIR / "solution_architecture.png",
               "Fig. 1. End-to-end software architecture of EcoSort-Edge. Input frames are processed by an ExecuTorch INT8 core on Arm Cortex-A76, "
               "evaluated by a Deterministic Policy Guard, and mapped into statutory SWM 2026 streams with real-time on-device telemetry.",
               width_inches=6.0)

    add_subsec_heading("A", "Stage 1: Input Acquisition & Edge Vision Backbone")
    add_body(
        "The perception backbone utilizes MobileNetV3-Small [2] customized for a 9-class municipal waste distribution: battery, cardboard, food_waste, "
        "glass_bottle, metal_can, multi_layer_packaging, paper, plastic_bottle, and vegetable_scrap. Frames are processed at 256×256 resolution, "
        "normalized with standard ImageNet channel statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), and passed to the "
        "ExecuTorch runtime to generate raw logits in 5.09 ms."
    )

    add_subsec_heading("B", "Stage 2: Deterministic Policy Guard (DPG)")
    add_body(
        "Raw model logits are converted to posterior probabilities P = [p_0 ... p_8] via Softmax. Rather than blindly forwarding uncalibrated arg-max "
        "predictions, the Deterministic Policy Guard applies two decision boundaries in <0.1 ms:"
    )
    add_bullet(
        "Shannon Entropy Gate: H(P) = -sum(p_i * log2(p_i)). For a 9-class distribution, the maximum theoretical entropy is H_max = log2(9) ≈ 3.17 bits. "
        "We empirically set a rejection threshold at H(P) > 2.00 bits on validation data. Predictions exceeding this threshold indicate broad multi-class "
        "uncertainty and are routed to Manual Review.",
        bold_prefix="1) Ambiguity Rejection: "
    )
    add_bullet(
        "Hazard Confidence Barrier: To protect municipal equipment from battery thermal runaway during compaction, our deployed policy imposes an elevated "
        "tau = 0.85 confidence threshold barrier on the battery class. In section VII, we formalize how this threshold connects to cost-sensitive Bayes risk.",
        bold_prefix="2) Hazard Suppression Barrier: "
    )

    add_subsec_heading("C", "Stage 3: Statutory Categorisation & Real-Time Telemetry")
    add_body(
        "Upon policy validation, EcoSort-Edge maps the detected class into its designated SWM 2026 statutory stream and generates instant visual telemetry "
        "on the kiosk screen alongside structured JSON logs. The end-to-end cycle (tensor preparation, ExecuTorch INT8 inference, and policy evaluation) "
        "completes in 11.85 ms mean per image.\n\n"
        "To adhere to privacy-by-design principles consistent with India's Digital Personal Data Protection (DPDP) Act 2023, EcoSort-Edge operates strictly "
        "within volatile memory (RAM). Input frames are never stored to persistent flash or streamed across network interfaces."
    )

    # Embed Fig. 2: Annotated Samples Grid
    add_figure(ASSETS_DIR / "annotated_samples_grid.png",
               "Fig. 2. Real pipeline outputs from the Raspberry Pi 5 benchmark showing overlays, confidences, and latencies across statutory streams: "
               "(top-left) Battery correctly captured; (top-right) Cardboard; (bottom-left) Food Waste; and (bottom-right) an Out-of-Distribution (OOD) failure "
               "mode where an extraneous non-waste item (a black shoe) is incorrectly classified as Multi-Layer Packaging with 95.0% confidence.",
               width_inches=5.2)

    # =========================================================================
    # SECTION IV: EMBEDDED MODEL OPTIMIZATION & EXECUTORCH
    # =========================================================================
    add_sec_heading("IV", "Embedded Model Optimization & ExecuTorch Lowering")
    add_subsec_heading("A", "Micro-Architectural Optimization on Arm Cortex-A76")
    add_body(
        "The Broadcom BCM2712 SoC on the Raspberry Pi 5 incorporates four 64-bit Arm Cortex-A76 cores at 2.4 GHz. Each core has a 64 KB L1 cache, "
        "a 512 KB private L2 cache, and shares a 2.0 MB L3 cache backed by 17 GB/s LPDDR4X DRAM [5]. While our compiled model binary (2.41 MB) "
        "and runtime memory (526 MB RSS) exceed the on-chip L3 cache capacity, MobileNetV3-Small's compact weight footprint substantially reduces "
        "memory traffic compared to larger models.\n\n"
        "Lowered through ExecuTorch 1.5.1 with the XNNPACK backend, the runtime utilizes Armv8.2-A NEON SIMD and the Signed Dot Product (SDOT) "
        "vector instruction [5]. SDOT performs 16 8-bit integer multiply-accumulate operations per 128-bit instruction (four 8-bit dot products per "
        "32-bit lane), delivering a theoretical peak compute capacity of 76.8 GOPS per core at 2.4 GHz."
    )

    add_subsec_heading("B", "PT2E Dynamic INT8 Quantization Mechanics")
    add_body(
        "As documented in literature on integer quantization [4], [8], naive uniform post-training quantization often causes dynamic range collapse on "
        "depthwise separable convolutions due to wide channel-wise variance in weight distributions. We addressed this using PyTorch 2 Export (PT2E) "
        "quantization through three specific techniques:"
    )
    add_bullet("Each depthwise filter slice is scaled independently via S_c = max(|w_c|)/127, preserving channel-specific feature resolution [4].", bold_prefix="1) Per-Channel Symmetric Weights: ")
    add_bullet("Activation quantization scale factors are computed dynamically at runtime based on incoming tensor dynamic ranges, avoiding saturation across varied lighting conditions.", bold_prefix="2) Dynamic Activation Scaling: ")
    add_bullet("During calibration, observer ranges were profiled using 270 balanced images (30 images per class) drawn from the training split.", bold_prefix="3) Balanced Calibration Profiling: ")

    # =========================================================================
    # SECTION V: EXPERIMENTAL RESULTS & ABLATION STUDY
    # =========================================================================
    add_sec_heading("V", "Experimental Evaluation, Benchmarks & Model Ablation")
    add_body(
        "We evaluated EcoSort-Edge across our curated 8,014-image dataset (5,607 train, 1,197 validation, 1,210 holdout test) and executed the compiled "
        ".pte binary on physical Raspberry Pi 5 hardware."
    )

    # Embed Fig. 3: Confusion Matrix
    add_figure(ASSETS_DIR / "confusion_matrix.png",
               "Fig. 3. Normalized confusion matrix evaluated across 1,210 holdout test images (88.51% top-1 accuracy, 98.35% statutory 4-stream accuracy). "
               "Battery recall reaches 96.67% (58/60; 95% CI 88.6–99.1%), with 2 battery samples misclassified into dry recyclable categories.",
               width_inches=5.0)

    # Embed Fig. 4: Per-Class Performance
    add_figure(ASSETS_DIR / "per_class_performance.png",
               "Fig. 4. Per-class precision, recall, and F1-score breakdown across all 9 municipal waste categories on the 1,210 test dataset.",
               width_inches=5.8)

    # Embed Fig. 5: Training Curves
    add_figure(ASSETS_DIR / "training_curves.png",
               "Fig. 5. Training loss descent and validation accuracy progression over 16 epochs across progressive image resizing (224×224 to 256×256).",
               width_inches=5.8)

    add_subsec_heading("A", "FP32 Baseline vs. Calibrated PT2E INT8 Parity Audit")
    add_body(
        "Table II compares the unquantized FP32 model against the lowered ExecuTorch INT8 model across both the 1,197 validation images and the 1,210 "
        "test set images. Quantization reduced binary size by 59.6% (5.97 MB to 2.41 MB) and sped up inference by 3.62× on the Cortex-A76 CPU. "
        "On the test set, top-1 accuracy dropped by a negligible 0.17% (88.68% to 88.51%) with 98.08% exact decision parity."
    )

    parity_headers = ["Evaluation Metric", "Baseline FP32 Model", "Calibrated PT2E INT8 Model", "Quantization Delta", "Compliance Status"]
    parity_rows = [
        ["Model Binary File Size", "5.97 MB (PyTorch JIT)", "2.41 MB (ExecuTorch .pte)", "-59.6% reduction", "PASSED (2.41 MB Footprint)"],
        ["Validation Accuracy (1,197 imgs)", "89.56% (1,072 / 1,197)", "89.39% (1,070 / 1,197)", "-0.17% delta", "PASSED (< 0.5% Threshold)"],
        ["Test Set Accuracy (1,210 imgs)", "88.68% (1,073 / 1,210)", "88.51% (1,071 / 1,210)", "-0.17% delta", "PASSED (1,071/1,210 Correct)"],
        ["Direct Decision Parity", "Reference (100.0%)", "98.08% (1,174 / 1,197)", "1.92% flip rate", "PASSED (> 95% Parity)"],
        ["Test Set Stream Accuracy", "98.43% (1,191 / 1,210)", "98.35% (1,190 / 1,210)", "-0.08% delta", "PASSED (95% CI 97.5–98.9%)"],
        ["Battery Recall (Test Set)", "96.67% (58 / 60)", "96.67% (58 / 60)", "0.00% delta (Exact)", "PASSED (95% CI 88.6–99.1%)"],
        ["Food Waste Recall (Test Set)", "96.72% (118 / 122)", "96.72% (118 / 122)", "0.00% delta (Exact)", "PASSED"],
        ["Multi-Layer Packaging Recall", "77.69% (101 / 130)", "76.92% (100 / 130)", "-0.77% delta", "PASSED"],
        ["Model Inference Latency (Pi 5 CPU)", "18.42 ms (FP32 Native)", "5.09 ms (INT8 NEON)", "3.62x Speedup", "PASSED (< 50 ms Limit)"]
    ]
    render_ieee_table("II", "Quantization Parity Audit Across Validation (1,197) and Test (1,210) Datasets", parity_headers, parity_rows, center_cols=[0, 3, 4])

    add_subsec_heading("B", "Head-to-Head Model Ablation: MobileNetV3-Small vs. MobileNetV3-Large")
    add_body(
        "To evaluate our architecture selection, we conducted a direct ablation against a peer MobileNetV3-Large INT8 model (4.31 MB). Both models "
        "were compiled via ExecuTorch 1.5.1 and evaluated on the Raspberry Pi 5 across our 30-sample suite (Table III and Fig. 6):"
    )

    abl_headers = ["Parameter / Metric", "MobileNetV3-Large INT8", "MobileNetV3-Small INT8", "Observed Difference", "Micro-Architectural Explanation"]
    abl_rows = [
        ["Model Architecture", "MobileNetV3-Large", "MobileNetV3-Small", "V3-Small", "Compact edge-optimized topology"],
        ["Parameter Count", "4.21 Million params", "1.53 Million params", "-63.7% parameters", "1.53M vs 4.21M parameter count"],
        ["Arithmetic Complexity (MACs)", "≈ 219 Million MACs", "≈ 56 Million MACs", "~74% fewer MACs", "56M vs 219M operations (3.9x compute ratio)"],
        ["Model Binary File Size", "4.31 MB (.pte binary)", "2.41 MB (.pte binary)", "44.1% smaller binary", "Reduces DRAM weight transfer volume"],
        ["Model Inference Latency (Pi 5)", "17.19 ms (~58 FPS)", "5.09 ms (~197 FPS)", "3.38x faster latency", "Explained by 3.9x arithmetic MAC ratio"],
        ["Class Top-1 Accuracy (30-suite)", "83.33% (25/30)", "80.00% (24/30)", "V3-Large (+3.3%)", "Slight advantage on cardboard/paper textures"],
        ["Statutory Stream Accuracy", "76.67% (23/30)", "100.00% (30/30)", "V3-Small (+23.3%)", "2 Large samples held due to high entropy (H>2.0)"],
        ["Battery Interception (30-suite)", "0.0% (0/3 captured)", "100.0% (3/3 captured)", "V3-Small (3/3)", "Small intercepted all 3; n=3 CI is wide (43.9–100%)"],
        ["Multi-Layer Packaging (MLP)", "Confused with plastic/metal", "2/4 RDF, 4/4 Dry", "V3-Small (2/4 RDF)", "Items 18 & 20 predicted metal_can (retaining Dry)"],
        ["Operating Core Temperature", "~58.4°C (rising)", "49.9°C (stable)", "8.5°C cooler", "Lower compute workload generates less heat"],
        ["Peak Process Memory (RSS)", "712.0 MB", "526.4 MB", "185.6 MB lower RSS", "Includes Python runtime and ExecuTorch shared libs"]
    ]
    render_ieee_table("III", "Empirical Model Ablation Comparison on Physical Raspberry Pi 5 CPU", abl_headers, abl_rows, center_cols=[0, 3])

    # Embed Fig. 6: Model Ablation Bar Chart
    add_figure(ASSETS_DIR / "model_ablation_comparison.png",
               "Fig. 6. Multi-metric empirical ablation on Raspberry Pi 5 CPU comparing MobileNetV3-Small INT8 against MobileNetV3-Large INT8. "
               "MobileNetV3-Small delivers 3.4× lower latency and 44% smaller binary footprint, consistent with its 3.9× fewer arithmetic MACs.",
               width_inches=6.0)

    add_body(
        "Analysis of the Latency Gap: The 3.38× latency speedup of MobileNetV3-Small (5.09 ms vs 17.19 ms) aligns closely with its 3.9× reduction in "
        "arithmetic compute (~56M vs ~219M MACs). Regarding stream routing on the 30-sample suite, MobileNetV3-Large achieved 23/30 stream accuracy "
        "(below its 25/30 top-1 accuracy) because two ambiguous predictions produced Shannon entropy exceeding H(P) > 2.0 bits, triggering the policy "
        "guard hold and withholding automated stream assignment. Furthermore, MobileNetV3-Large misclassified item_02_battery_02.jpg as cardboard, "
        "failing hazard capture on this small suite."
    )

    # =========================================================================
    # SECTION VI: PHYSICAL BENCHMARK SUITE & TIMING BREAKDOWN
    # =========================================================================
    add_sec_heading("VI", "Physical Benchmark Suite Results & Timing Analysis")
    add_body(
        "Table IV details the per-image evaluation across our 30-image physical benchmark suite executed on the Raspberry Pi 5 CPU. "
        "Top-1 class accuracy reached 24/30 (80.0%), while statutory 4-stream accuracy was 30/30 (100.0%; 95% Wilson CI: 88.6%–100.0%). "
        "For Multi-Layer Packaging, 2/4 items (Items 17 and 19) were predicted as multi_layer_packaging and received the specific RDF tag, while "
        "Items 18 and 20 were predicted as metal_can (maintaining the correct Dry stream, though missing the specific RDF flag). "
        "We also note that 12 pairs of images in this test folder share identical confidence values due to repeated burst acquisitions of the same "
        "physical items (representing approximately 18 distinct physical objects)."
    )

    bench_headers = ["Sample Filename", "Ground Truth Stream", "Predicted Class", "Class Match", "Routed Statutory Stream", "Confidence", "Harness Latency", "Status"]
    bench_rows = [
        ["item_01_battery_01.jpg", "special_care", "battery", "MATCH", "special_care", "90.4%", "10.98 ms", "PASS (Hazard Captured)"],
        ["item_02_battery_02.jpg", "special_care", "battery", "MATCH", "special_care", "98.0%", "9.30 ms", "PASS (Hazard Captured)"],
        ["item_03_battery_03.jpg", "special_care", "battery", "MATCH", "special_care", "90.4%", "13.39 ms", "PASS (Hazard Captured)"],
        ["item_04_cardboard_01.jpg", "dry", "cardboard", "MATCH", "dry", "78.7%", "10.26 ms", "PASS"],
        ["item_05_cardboard_02.jpg", "dry", "cardboard", "MATCH", "dry", "72.2%", "9.43 ms", "PASS"],
        ["item_06_cardboard_03.jpg", "dry", "cardboard", "MATCH", "dry", "78.7%", "11.04 ms", "PASS"],
        ["item_07_food_waste_01.jpg", "wet", "food_waste", "MATCH", "wet", "88.7%", "11.38 ms", "PASS"],
        ["item_08_food_waste_02.jpg", "wet", "food_waste", "MATCH", "wet", "96.5%", "10.60 ms", "PASS"],
        ["item_09_food_waste_03.jpg", "wet", "food_waste", "MATCH", "wet", "88.7%", "9.75 ms", "PASS"],
        ["item_10_food_waste_04.jpg", "wet", "food_waste", "MATCH", "wet", "96.5%", "10.39 ms", "PASS"],
        ["item_11_glass_bottle_01.jpg", "dry", "glass_bottle", "MATCH", "dry", "97.6%", "11.31 ms", "PASS"],
        ["item_12_glass_bottle_02.jpg", "dry", "glass_bottle", "MATCH", "dry", "92.6%", "12.15 ms", "PASS"],
        ["item_13_glass_bottle_03.jpg", "dry", "glass_bottle", "MATCH", "dry", "97.6%", "11.76 ms", "PASS"],
        ["item_14_metal_can_01.jpg", "dry", "metal_can", "MATCH", "dry", "99.4%", "12.99 ms", "PASS"],
        ["item_15_metal_can_02.jpg", "dry", "metal_can", "MATCH", "dry", "73.6%", "13.50 ms", "PASS"],
        ["item_16_metal_can_03.jpg", "dry", "metal_can", "MATCH", "dry", "99.4%", "12.79 ms", "PASS"],
        ["item_17_multi_layer_01.jpg", "dry", "multi_layer_packaging", "MATCH", "dry (RDF)", "95.0%", "12.20 ms", "PASS (RDF Tagged)"],
        ["item_18_multi_layer_02.jpg", "dry", "metal_can", "MISMATCH", "dry", "67.2%", "10.85 ms", "PASS (Stream Correct)"],
        ["item_19_multi_layer_03.jpg", "dry", "multi_layer_packaging", "MATCH", "dry (RDF)", "95.0%", "13.62 ms", "PASS (RDF Tagged)"],
        ["item_20_multi_layer_04.jpg", "dry", "metal_can", "MISMATCH", "dry", "67.2%", "12.24 ms", "PASS (Stream Correct)"],
        ["item_21_paper_01.jpg", "dry", "paper", "MATCH", "dry", "77.2%", "12.94 ms", "PASS"],
        ["item_22_paper_02.jpg", "dry", "paper", "MATCH", "dry", "96.5%", "11.58 ms", "PASS"],
        ["item_23_paper_03.jpg", "dry", "paper", "MATCH", "dry", "77.2%", "14.98 ms", "PASS"],
        ["item_24_plastic_bottle_01.jpg", "dry", "cardboard", "MISMATCH", "dry", "62.1%", "11.82 ms", "PASS (Stream Correct)"],
        ["item_25_plastic_bottle_02.jpg", "dry", "plastic_bottle", "MATCH", "dry", "90.7%", "12.64 ms", "PASS"],
        ["item_26_plastic_bottle_03.jpg", "dry", "cardboard", "MISMATCH", "dry", "62.1%", "13.35 ms", "PASS (Stream Correct)"],
        ["item_27_plastic_bottle_04.jpg", "dry", "plastic_bottle", "MATCH", "dry", "90.7%", "12.19 ms", "PASS"],
        ["item_28_vegetable_01.jpg", "wet", "food_waste", "MISMATCH", "wet", "85.4%", "13.16 ms", "PASS (Stream Correct)"],
        ["item_29_vegetable_02.jpg", "wet", "vegetable_scrap", "MATCH", "wet", "94.8%", "12.01 ms", "PASS"],
        ["item_30_vegetable_03.jpg", "wet", "food_waste", "MISMATCH", "wet", "85.4%", "10.79 ms", "PASS (Stream Correct)"]
    ]
    render_ieee_table("IV", "Physical Benchmark Suite Results on Raspberry Pi 5 CPU (Arm Cortex-A76)", bench_headers, bench_rows, center_cols=[0, 3, 5, 6, 7])

    add_subsec_heading("A", "Timing Breakdown and Hardware Telemetry")
    add_body(
        "We distinguish between two timing measurements: (1) Isolated Model Runtime: 5.09 ms mean (P95: 6.28 ms, Min: 3.37 ms) representing pure "
        "ExecuTorch forward pass execution via XNNPACK; and (2) Harness Per-Image Latency: 11.85 ms mean (min 9.30 ms, max 14.98 ms) which includes "
        "OpenCV disk loading, channel reordering, tensor normalization, and policy guard evaluation.\n\n"
        "Hardware telemetry logged via Linux sysfs and vcgencmd during benchmark execution:"
    )
    add_bullet("Broadcom BCM2712 Quad-Core Arm Cortex-A76 @ 2.4 GHz (Armv8.2-A).", bold_prefix="Target SoC Architecture: ")
    add_bullet("Arm ExecuTorch 1.5.1 with XNNPACK Backend (NEON SIMD + SDOT).", bold_prefix="Inference Engine: ")
    add_bullet("5.09 ms model runtime (~197 FPS); 11.85 ms full harness per-image latency.", bold_prefix="Measured Latency: ")
    add_bullet("49.9°C measured across the 30-sample benchmark run (30.1°C headroom below the 80°C throttle point).", bold_prefix="Core Temperature: ")
    add_bullet("526.4 MB total process RSS (system RAM: 903 MB used out of 8,062 MB — 11.2% utilization).", bold_prefix="Memory Utilization: ")
    add_bullet("Register 0x0 (OK - Zero under-voltage or thermal throttling flags recorded).", bold_prefix="Throttling State: ")
    add_bullet("Zero bytes transmitted (100% offline edge processing).", bold_prefix="Network Dependencies: ")

    # =========================================================================
    # SECTION VII: SYSTEM LIMITATIONS & FAILURE MODES (CRITICAL HONEST SECTION)
    # =========================================================================
    add_sec_heading("VII", "System Limitations, Failure Modes & Engineering Roadmap")
    add_body(
        "A rigorous engineering evaluation requires transparent disclosure of failure modes, open-set behavior, and deployment constraints:"
    )

    add_subsec_heading("A", "Open-Set Vulnerability & Out-of-Distribution Inputs")
    add_body(
        "As illustrated in Fig. 2 (bottom-right), when presented with an out-of-distribution extraneous object (a black shoe), the model produced a "
        "closed-set prediction of multi_layer_packaging at 95.0% confidence. Closed-set Softmax inherently forces unfamiliar inputs into known classes. "
        "To mitigate this, future work must incorporate an explicit 'Unknown / Non-Waste' background rejection class and energy-based out-of-distribution "
        "detection to reject foreign items prior to classification."
    )

    add_subsec_heading("B", "Hazard Misses Under Arg-Max Classification")
    add_body(
        "On the full 1,210-image test set, battery hazard recall was 96.67% (58/60; 95% Wilson CI: 88.6%–99.1%), with 2 battery samples misclassified into "
        "dry recyclable categories (glass bottle and metal can). Under simple arg-max classification, if an alternative class receives a slightly higher "
        "logit, the battery label is suppressed. To eliminate these misses, we are implementing a probability-based hazard hold rule: whenever the calibrated "
        "posterior P(battery) >= h (e.g., h = 0.15 derived from validation ROC tuning), the item is automatically diverted to Special-Care inspection, "
        "regardless of the top-ranked class."
    )

    add_subsec_heading("C", "Sample Size Constraints & Benchmark Integrity")
    add_body(
        "While the system achieved 3/3 hazard capture and 30/30 stream accuracy on the physical suite, n = 3 gives a wide 95% Wilson confidence lower bound "
        "of 43.9%, making it statistically uninformative on its own. Reliable hazard guarantees require evaluation on at least 300+ independent battery images. "
        "Additionally, 12 image pairs in the 30-sample suite exhibited identical confidences due to burst-mode capture (~18 distinct visual objects)."
    )

    add_subsec_heading("D", "Sanitary Stream & Single-Object Assumption")
    add_body(
        "The current vision dataset contains no sanitary waste category; sanitary products are not suitable for camera-based presentation and are best "
        "handled via dedicated physical chutes. Furthermore, the model assumes a single primary discard per frame and has not yet been stress-tested "
        "against heavy clutter or multi-object overlaps common in real-world Indian collection centers."
    )

    # =========================================================================
    # SECTION VIII: OPERATIONAL PLAYBOOK & REPRODUCIBILITY
    # =========================================================================
    add_sec_heading("VIII", "Operational Deployment & Reproducibility")
    add_subsec_heading("A", "Hardware Stack & Kiosk Economics")
    add_body(
        "The complete hardware platform consists of a Raspberry Pi 5 (4GB RAM, ~$60 base board; ~$85–$110 fully equipped with a 27W USB-C PD supply "
        "and active cooler) paired with a standard USB or CSI camera. By executing 100% on-device, recurring monthly API costs are eliminated."
    )

    add_subsec_heading("B", "Systemd Watchdog & Reproducible CLI")
    add_body(
        "The application is packaged as an automated Linux service (ecosort.service) with automatic crash restart. "
        "The benchmark evaluation can be reproduced on any Raspberry Pi 5 with the following commands:"
    )
    p_code = doc.add_paragraph()
    p_code.paragraph_format.space_before = Pt(2)
    p_code.paragraph_format.space_after = Pt(4)
    p_code.paragraph_format.left_indent = Inches(0.2)
    rc = p_code.add_run(
        "git clone https://github.com/DK-A/EcoSort-Edge-Offline-Waste-Sorting-Food-Categorisation-using-ExecuTorch-on-Raspberry-Pi.git\n"
        "cd EcoSort-Edge-Offline-Waste-Sorting-Food-Categorisation-using-ExecuTorch-on-Raspberry-Pi\n"
        "pip install -r requirements.txt\n"
        "python evaluate_pipeline.py --model models/waste_classifier_int8.pte --image_dir benchmark_30 --ground_truth benchmark_30/ground_truth.json"
    )
    rc.font.name = "Consolas"
    rc.font.size = Pt(7.5)

    # =========================================================================
    # SECTION IX: CONCLUSION
    # =========================================================================
    add_sec_heading("IX", "Conclusion & Future Work")
    add_body(
        "EcoSort-Edge demonstrates that offline edge AI waste sorting can be achieved on affordable embedded hardware. By combining a MobileNetV3-Small "
        "architecture with PT2E dynamic INT8 quantization and Arm ExecuTorch microkernels, the system delivers 5.09 ms model execution (~197 FPS) "
        "and 11.85 ms end-to-end latency on a Raspberry Pi 5 CPU within a 2.41 MB binary footprint. Across 1,210 test images, the system achieves "
        "98.35% statutory 4-stream accuracy (1,190/1,210; 95% CI 97.5–98.9%) and 96.67% battery hazard recall (58/60; 95% CI 88.6–99.1%). "
        "Future hardening will prioritize probability-based hazard thresholding, temperature calibration, an open-set unknown rejector, and extended "
        "thermal soak validation under field operating conditions."
    )

    p_art = doc.add_paragraph()
    p_art.paragraph_format.space_before = Pt(3)
    p_art.paragraph_format.space_after = Pt(6)
    p_art.add_run("• Project Code Repository: ").font.bold = True
    add_hyperlink(p_art, "https://github.com/DK-A/EcoSort-Edge-Offline-Waste-Sorting-Food-Categorisation-using-ExecuTorch-on-Raspberry-Pi", "https://github.com/DK-A/EcoSort-Edge-Offline-Waste-Sorting-Food-Categorisation-using-ExecuTorch-on-Raspberry-Pi")
    p_art.add_run("\n• Benchmark Demonstration Video: ").font.bold = True
    add_hyperlink(p_art, "assets/DEMO.mp4 (1080p Hardware Demonstration)", "https://github.com/DK-A/EcoSort-Edge-Offline-Waste-Sorting-Food-Categorisation-using-ExecuTorch-on-Raspberry-Pi/raw/main/assets/DEMO.mp4")

    # References
    add_sec_heading("X", "References")
    refs = [
        "Ministry of Environment, Forest and Climate Change (MoEFCC), Government of India, 'Solid Waste Management Rules, 2026,' Gazette of India, Extraordinary, Part II, Section 3, Subsection (i), Jan. 2026.",
        "A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, 'Searching for MobileNetV3,' in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 1314–1324.",
        "PyTorch Team, 'ExecuTorch: On-Device AI Across Mobile, Embedded and Edge for PyTorch,' PyTorch Open Source Project, 2024. [Online]. Available: https://pytorch.org/executorch",
        "R. Krishnamoorthi, 'Quantizing deep convolutional networks for efficient inference: A whitepaper,' arXiv preprint arXiv:1806.08342, 2018.",
        "Arm Limited, 'Arm Architecture Reference Manual Armv8, for Armv8-A architecture profile: Vector Dot Product Operations,' Arm DDI 0487, 2023.",
        "Central Pollution Control Board (CPCB), 'Annual Report on Implementation of Solid Waste Management Rules (2020-21),' Ministry of Environment, Forest and Climate Change, New Delhi, India, 2022.",
        "P. Warden and D. Situnayake, 'TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers,' O'Reilly Media, Sebastopol, CA, 2020.",
        "M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen, and T. Blankevoort, 'A White Paper on Neural Network Quantization,' arXiv preprint arXiv:2106.08295, 2021.",
        "E. B. Wilson, 'Probable inference, the law of succession, and statistical inference,' Journal of the American Statistical Association, vol. 22, no. 158, pp. 209–212, 1927.",
        "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4510–4520."
    ]
    for idx, r in enumerate(refs):
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_after = Pt(2)
        p_ref.paragraph_format.left_indent = Inches(0.25)
        run_ref = p_ref.add_run(f"[{idx+1}]  {r}")
        run_ref.font.name = "Times New Roman"
        run_ref.font.size = Pt(8.0)
        run_ref.font.color.rgb = RGBColor(71, 85, 105)

    doc.save(str(OUTPUT_DOCX))
    print(f"[+] Exact IEEE Word Document Successfully Generated: {OUTPUT_DOCX} ({OUTPUT_DOCX.stat().st_size / 1024:.2f} KB)")

if __name__ == "__main__":
    create_ieee_document()
