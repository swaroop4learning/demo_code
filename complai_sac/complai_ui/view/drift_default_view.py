from view import drift_multi_class_view, drift_bin_reg_view
from service.drift_service import DriftService

def app(model):
    driftObj = DriftService(model)
    problem_type =driftObj.problem_type
    if(problem_type=='multiclass'):
        drift_multi_class_view.app(model)
    else:
        drift_bin_reg_view.app(model)