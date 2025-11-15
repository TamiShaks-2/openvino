import os
import pytest
import numpy as np
import torch
import openvino as ov 
from openvino.frontend import FrontEndManager
from openvino import PartialShape 

# יצירת מנהל Frontend Manager גלובלי (מאפשר גישה למתורגמנים השונים)
fe_manager = FrontEndManager()

# ----------------------------------------------------------------------------
# 1. PyTest Fixtures 
# ----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def core():
    """יוצר אובייקט Core של OpenVINO פעם אחת לכל הטסטים."""
    return ov.Core()

# ----------------------------------------------------------------------------
# 2. מודל PyTorch ופונקציות עזר
# ----------------------------------------------------------------------------

class MultiDotModel(torch.nn.Module):
    """מודל PyTorch פשוט שעוטף את torch.linalg.multi_dot."""
    def forward(self, *tensors):
        return torch.linalg.multi_dot(tensors)

def get_precision_settings(precision_str):
    """מחזיר סוגי נתונים וטולרנסים לבדיקה."""
    if precision_str == "FP16":
        return torch.float16, np.float16, {"rtol": 1e-3, "atol": 1e-3}
    return torch.float32, np.float32, {"rtol": 1e-5, "atol": 1e-5}

def run_test_and_compare(ie_device, precision, core, input_shapes):
    """
    פונקציית העזר המרכזית: מריצה PT, ממירה ל-OV, מריצה OV ומשווה תוצאות.
    """
    torch_dtype, np_dtype, tolerance = get_precision_settings(precision)

    # 1. יצירת קלט
    np.random.seed(42)
    np_inputs = [np.random.rand(*shape).astype(np_dtype) for shape in input_shapes]
    torch_inputs = [torch.tensor(arr, dtype=torch_dtype) for arr in np_inputs]
    
    # 2. הרצת PyTorch (Reference)
    model_pt = MultiDotModel()
    model_pt.eval()
    with torch.no_grad():
        result_pt = model_pt(*torch_inputs).numpy()

    # 3. המרת OpenVINO 
    example_input_tuple = tuple(torch_inputs)
    model_ov = None
    
    try:
        # נסה את ה-API המודרני הכללי (ov.convert_model)
        model_ov = ov.convert_model(model_pt, example_input=example_input_tuple)
    except (AttributeError, Exception) as e:
        # 💡 תיקון: אם ov.convert_model לא קיים או נכשל, השתמש ב-PyTorch frontend (Fallback)
        try:
            fe = fe_manager.load_by_name("pytorch")
            # אם fe הוא None, משמעות הדבר היא שלא נטען FrontEnd מתאים, וזה יביא ל-AttributeError ב-fe.convert.
            if fe is None:
                 raise RuntimeError(f"לא ניתן לטעון את FrontEnd 'pytorch'. שגיאה קודמת: {e}")
            model_ov = fe.convert(model_pt, example_input=example_input_tuple)
        except Exception as fe_e:
             # אם ה-Fallback נכשל, צור כשל ב-pytest
             pytest.fail(f"המרת המודל נכשלה באמצעות FrontEnd API: {fe_e} (שגיאה ראשונית: {e})")

    # 4. הרצת OpenVINO
    compiled_model = core.compile_model(model_ov, ie_device)
    infer_request = compiled_model.create_infer_request()
    
    ov_inputs = {}
    for i, (param, data) in enumerate(zip(compiled_model.inputs, np_inputs)):
        ov_inputs[param.get_any_name()] = data
        
    result_ov = infer_request.infer(ov_inputs)[compiled_model.output(0)]

    # 5. השוואה נומרית
    np.testing.assert_allclose(result_pt, result_ov, **tolerance)
    
    # 6. החזרת המודל לבדיקות גרף
    return model_ov

# ----------------------------------------------------------------------------
# 3. בדיקות נכונות נומרית (Class 1)
# ----------------------------------------------------------------------------

class TestNumericalCorrectness:
    """
    בודק שהתוצאות של OV זהות ל-PT עבור תרחישים שונים.
    """

    @pytest.mark.parametrize("shapes", [
        pytest.param([(2, 5), (5, 3)], id="N2_basic"),
        pytest.param([(2, 100), (100, 2), (2, 100)], id="N3_left_optimal"),
        pytest.param([(100, 2), (2, 100), (100, 2)], id="N3_right_optimal"),
        pytest.param([(10, 2), (2, 50), (50, 20), (20, 5)], id="N4_dp_optimal"),
        pytest.param([(50, 5), (5, 10), (10, 20), (20, 2)], id="N4_dp_another"),
        pytest.param([(10, 5), (5, 2), (2, 20), (20, 10), (10, 5)], id="N5_dp_complex"),
    ])
    @pytest.mark.parametrize("ie_device", ["CPU"])
    @pytest.mark.parametrize("precision", ["FP32", "FP16"])
    def test_matrix_cases(self, ie_device, precision, core, shapes):
        """בדיקת מקרים של מטריצה כפול מטריצה."""
        if ie_device == "GPU": 
            pytest.skip("Skipping GPU test as requested.")
        run_test_and_compare(ie_device, precision, core, shapes)

    @pytest.mark.parametrize("shapes", [
        pytest.param([(5,), (5, 10), (10,)], id="N3_vec_mat_vec"),
        pytest.param([(5,), (5, 10), (10, 3)], id="N3_vec_mat_mat"),
        pytest.param([(3, 5), (5, 10), (10,)], id="N3_mat_mat_vec"),
    ])
    @pytest.mark.parametrize("ie_device", ["CPU"])
    @pytest.mark.parametrize("precision", ["FP32", "FP16"])
    def test_1d_endpoint_cases(self, ie_device, precision, core, shapes):
        """
        בודק את ההתנהגות המיוחדת של 1D בקצוות
        """
        if ie_device == "GPU":
            pytest.skip("Skipping GPU test as requested.")
        run_test_and_compare(ie_device, precision, core, shapes)

# ----------------------------------------------------------------------------
# 4. בדיקות אופטימליות גרף (Class 2)
# ----------------------------------------------------------------------------

def _get_graph_str_recursive(node, param_names):
    """
    פונקציית עזר רקורסיבית לבניית ייצוג מחרוזתי של סדר הפעולות.
    """
    op_type = node.get_type_name()
    
    if op_type == "Parameter":
        return param_names.get(node.get_friendly_name(), "?P?")

    if op_type == "MatMul":
        left = _get_graph_str_recursive(node.input_value(0).get_node(), param_names)
        right = _get_graph_str_recursive(node.input_value(1).get_node(), param_names)
        return f"({left}@{right})"
        
    if op_type in ("Convert", "Unsqueeze", "Squeeze"):
        return _get_graph_str_recursive(node.input_value(0).get_node(), param_names)
        
    return f"[{op_type}]"

def get_matmul_graph_structure(ov_model):
    """
    מחזיר מחרוזת המייצגת את סדר ה-MatMul בגרף.
    """
    params = ov_model.get_parameters()
    param_names = {p.get_friendly_name(): f"P{i}" for i, p in enumerate(params)}
    
    output_node = ov_model.output(0).get_node()
    
    # חפש אחורה את הפעולה הראשית לפני Squeeze אם קיימת
    while output_node.get_type_name() == "Squeeze":
        output_node = output_node.input_value(0).get_node()
        
    return _get_graph_str_recursive(output_node, param_names)

class TestGraphOptimality:
    """
    בודק את מבנה הגרף שנוצר כדי לוודא שהאופטימיזציה
    (היוריסטיקה / DP) התרחשה כצפוי.
    """
    
    @pytest.mark.parametrize("shapes, expected_graph_str", [
        pytest.param([(2, 100), (100, 2), (2, 100)], "((P0@P1)@P2)", id="N3_left_optimal"),
        pytest.param([(100, 2), (2, 100), (100, 2)], "(P0@(P1@P2))", id="N3_right_optimal"),
        pytest.param([(10, 2), (2, 50), (50, 20), (20, 5)], "(P0@((P1@P2)@P3))", id="N4_dp_optimal"),
    ])
    @pytest.mark.parametrize("ie_device", ["CPU"])
    @pytest.mark.parametrize("precision", ["FP32", "FP16"])
    def test_static_shape_optimality(self, ie_device, precision, core, shapes, expected_graph_str):
        """
        בודק שהגרף שנוצר עבור צורות סטטיות תואם את הסדר האופטימלי הצפוי.
        """
        if ie_device == "GPU":
            pytest.skip("Skipping GPU test as requested.")
            
        model_ov = run_test_and_compare(ie_device, precision, core, shapes)
        
        graph_str = get_matmul_graph_structure(model_ov)
        assert graph_str == expected_graph_str

    @pytest.mark.parametrize("ie_device", ["CPU"])
    @pytest.mark.parametrize("precision", ["FP32", "FP16"])
    def test_dynamic_shape_fallback(self, ie_device, precision, core):
        """
        בודק שבמקרה של צורות דינמיות, המתרגם חוזר ל"מצב הבטוח"
        של שרשור שמאלי (left-associative chain).
        """
        if ie_device == "GPU":
            pytest.skip("Skipping GPU test as requested.")

        model_pt = MultiDotModel()
        
        dynamic_inputs = [
            PartialShape([-1, -1]), # P0
            PartialShape([-1, -1]), # P1
            PartialShape([-1, -1]), # P2
            PartialShape([-1, -1]), # P3
        ]
        
        model_ov = None
        # 💡 תיקון: שימוש ב-ov.convert_model עם fallback ל-PyTorch frontend וטיפול בשגיאות
        try:
            model_ov = ov.convert_model(model_pt, input=dynamic_inputs)
        except (AttributeError, Exception) as e:
            try:
                fe = fe_manager.load_by_name("pytorch")
                if fe is None:
                    raise RuntimeError(f"לא ניתן לטעון את FrontEnd 'pytorch'. שגיאה קודמת: {e}")
                model_ov = fe.convert(model_pt, input=dynamic_inputs)
            except Exception as fe_e:
                pytest.fail(f"המרת המודל הדינמי נכשלה באמצעות FrontEnd API: {fe_e} (שגיאה ראשונית: {e})")

        # סדר הקלט הדינמי אמור לחזור לשרשור שמאלי 
        expected_fallback_str = "(((P0@P1)@P2)@P3)"
        graph_str = get_matmul_graph_structure(model_ov)
        
        assert graph_str == expected_fallback_str