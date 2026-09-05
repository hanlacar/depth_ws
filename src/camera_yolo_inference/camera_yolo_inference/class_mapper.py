class SemanticClassMapper:
    def __init__(self,manifest):self.manifest=manifest;self.mapping={};self.warnings=[]
    def resolve_model_classes(self,model_names):
        names=dict(enumerate(model_names)) if isinstance(model_names,(list,tuple)) else {int(k):v for k,v in model_names.items()}
        if len(names.values())!=len(set(names.values())):raise ValueError("duplicate model class names")
        claimed={};mapping={}
        for role,spec in self.manifest["semantic_roles"].items():
            aliases=set(spec["accepted_dataset_names"]);ids=[class_id for class_id,name in names.items() if name in aliases]
            for class_id in ids:
                if class_id in claimed:raise ValueError(f"class {class_id} maps to two roles")
                claimed[class_id]=role
            mapping[role]=ids
        unclaimed=[f"{class_id}:{name}" for class_id,name in names.items() if class_id not in claimed]
        self.warnings=[] if not unclaimed else [
            "ignored model classes absent from manifest aliases: "+", ".join(unclaimed)]
        self.mapping=mapping;self.validate_required_roles();return mapping
    def validate_required_roles(self):
        for role,spec in self.manifest["semantic_roles"].items():
            if spec.get("required") and not self.mapping.get(role):raise ValueError(f"required role has no model class: {role}")
    def class_ids_for_role(self,role):return tuple(self.mapping.get(role,()))
    def describe_mapping(self):return {role:list(ids) for role,ids in self.mapping.items()}
