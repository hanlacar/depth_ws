import unittest

from camera_yolo_inference.class_mapper import SemanticClassMapper


def manifest(required=True):
    return {"semantic_roles": {
        "road": {"required": required, "accepted_dataset_names": ["road"]},
        "white_line": {"required": required, "accepted_dataset_names": ["W_line"]},
        "yellow_line": {"required": required, "accepted_dataset_names": ["Y_line"]},
        "red_light": {"required": required, "accepted_dataset_names": ["R_light"]},
        "green_light": {"required": required, "accepted_dataset_names": ["G_light"]},
    }}


class ModelSemanticContractTest(unittest.TestCase):
    def test_class_order_is_irrelevant_and_y_light_is_not_required(self):
        names = {9: "G_light", 2: "road", 7: "R_light", 0: "Y_line", 4: "W_line"}
        mapping = SemanticClassMapper(manifest()).resolve_model_classes(names)
        self.assertEqual(mapping["road"], [2])
        self.assertNotIn("Y_light", names.values())

    def test_missing_required_class_fails(self):
        with self.assertRaisesRegex(ValueError, "green_light"):
            SemanticClassMapper(manifest()).resolve_model_classes(
                ["road", "W_line", "Y_line", "R_light"])

    def test_extra_class_is_warned_and_ignored(self):
        mapper = SemanticClassMapper(manifest())
        mapper.resolve_model_classes(
            ["road", "W_line", "Y_line", "R_light", "G_light", "new_optional"])
        self.assertTrue(mapper.warnings)
        self.assertIn("new_optional", mapper.warnings[0])

    def test_duplicate_semantic_claim_is_rejected(self):
        duplicate = manifest()
        duplicate["semantic_roles"]["also_road"] = {
            "required": False, "accepted_dataset_names": ["road"]}
        with self.assertRaisesRegex(ValueError, "two roles"):
            SemanticClassMapper(duplicate).resolve_model_classes(
                ["road", "W_line", "Y_line", "R_light", "G_light"])


if __name__ == "__main__":
    unittest.main()
