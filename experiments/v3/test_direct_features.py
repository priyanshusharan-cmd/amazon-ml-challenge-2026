"""Meaningful role/absence/invariance regression tests; no datasets are required."""
import unittest

import numpy as np
import polars as pl

from direct_features import FEATURES, INPUTS, add_features, address_parts


def pair(q_addr="12 Main Street, Apt 4", s_addr="12 Main St Suite 4", **kw):
    values = dict(q_idx=[3], s1_idx=[7], q_name=["Acme Ltd"], s_name=["ACME LTD"],
                  q_addr=[q_addr], s_addr=[s_addr], q_core=["acme"], s_core=["acme"],
                  q_anorm=["12 main st 4"], s_anorm=["12 main st 4"])
    values.update({k: [v] for k, v in kw.items()})
    return pl.DataFrame(values)


class AddressRolesTest(unittest.TestCase):
    def test_unit_labels_and_leading_zero_canonicalization(self):
        p = address_parts("0012 Main Street, apartment 004, Floor 02")
        self.assertEqual((p.premise, p.units, p.floors), ("12", ("4",), ("2",)))

    def test_floor_does_not_become_house(self):
        p = address_parts("3rd floor, 12 Main Street")
        self.assertEqual(p.floors, ("3",))
        self.assertEqual(p.premise, "")  # later unlabeled numbers remain conservative
        self.assertEqual(address_parts("3rd Avenue 90210").premise, "")

    def test_french_explicit_roles(self):
        p = address_parts("N°12 bis rue des Fleurs, appartement 4, 2ème étage")
        self.assertEqual((p.premise, p.units, p.floors), ("12bis", ("4",), ("2",)))

    def test_compound_house_and_suffixes_survive(self):
        self.assertEqual(address_parts("D.No.1-04-283/3 Main Road").premise, "1-4-283/3")
        self.assertEqual(address_parts("12B Main Street").premise, "12b")

    def test_later_postcode_is_not_a_premise(self):
        self.assertEqual(address_parts("Main Street, Springfield, 90210").premise, "")

    def test_hash_after_street_is_unit(self):
        self.assertEqual(address_parts("#12 Main St #4").units, ("4",))
        self.assertEqual(address_parts("#12 Main St #4").premise, "12")

    def test_letter_unit_and_unicode_digits(self):
        self.assertEqual(address_parts("１２ Main St Suite B").units, ("b",))
        self.assertEqual(address_parts("१२ Main St Unit No 4").premise, "12")
        self.assertEqual(address_parts("१२ Main St Unit No 4").units, ("4",))

    def test_label_belongs_to_following_identifier(self):
        p = address_parts("12 Apt 4 Main Street")
        self.assertEqual((p.premise, p.units), ("12", ("4",)))
        p = address_parts("12 Floor 4 Main Street")
        self.assertEqual((p.premise, p.floors), ("12", ("4",)))


class PairFeaturesTest(unittest.TestCase):
    def test_equal_house_unit_and_alpha_address(self):
        d = add_features(pair()).row(0, named=True)
        self.assertEqual((d["d3_prem_equal"], d["d3_unit_equal"], d["d3_addr_alpha_ratio"]), (1, 1, 1))

    def test_role_swap_is_detected(self):
        d = add_features(pair("12 Main St Unit 4", "4 Main St Unit 12")).row(0, named=True)
        self.assertEqual((d["d3_prem_conflict"], d["d3_unit_conflict"], d["d3_role_cross_match"]), (1, 1, 1))

    def test_suffix_conflict_survives_normalized_number_equality(self):
        d = add_features(pair("12A Main St", "12B Main St")).row(0, named=True)
        self.assertEqual((d["d3_prem_equal"], d["d3_prem_suffix_conflict"], d["d3_prem_numset_jaccard"]), (0, 1, 1))

    def test_missing_values_do_not_create_positive_evidence(self):
        d = pair(**{k: None for k in INPUTS})
        x = add_features(d).select(FEATURES).to_numpy()
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(float(x.sum()), 0.0)

    def test_typos_differ_from_changed_street_and_name(self):
        typo = add_features(pair(q_anorm="12 maim st", q_core="acne")).row(0, named=True)
        changed = add_features(pair(q_anorm="12 elm st", q_core="beta")).row(0, named=True)
        self.assertGreater(typo["d3_addr_qonly_fuzzy"], changed["d3_addr_qonly_fuzzy"])
        self.assertGreater(typo["d3_name_qonly_fuzzy"], changed["d3_name_qonly_fuzzy"])

    def test_duplication_and_batching_do_not_change_features(self):
        d = pl.concat([pair(), pair("14 Elm Rd Unit B", "12 Main St Unit 4")])
        whole = add_features(d)
        self.assertTrue(whole.equals(pl.concat([add_features(d.head(1)), add_features(d.tail(1))])))
        duplicated = add_features(pl.concat([d, d.head(1)]))
        self.assertTrue(whole.equals(duplicated.head(2)))

    def test_empty_schema_and_dtypes(self):
        d = add_features(pair().head(0))
        self.assertEqual(d.columns, ["q_idx", "s1_idx", *FEATURES])
        self.assertTrue(all(d.schema[c] == pl.Float32 for c in FEATURES))

    def test_acronym_requires_multiple_words_and_actual_match(self):
        d = add_features(pair(q_core="ibm", s_core="international business machines")).row(0, named=True)
        self.assertEqual(d["d3_name_acronym_equal"], 1)
        self.assertEqual(add_features(pair()).item(0, "d3_name_acronym_equal"), 0)


if __name__ == "__main__":
    unittest.main()
