import os
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import pandas as pd


def create_remap(series: pd.Series) -> tuple[dict, np.ndarray]:
    """
    Return two mappers: from ids to the smallest integer set and back (one dict and one ndarray for speed)
    :param series: pd.Series with unique ids
    :return: (forward_dict, reverse_ndarray)
    """
    assert series.is_unique, "Series must have unique values"
    arr = np.sort(series.unique())
    return {val: i for i, val in enumerate(arr)}, arr


@dataclass
class Data:
    reviews: pd.DataFrame
    users: pd.DataFrame
    organisations: pd.DataFrame
    features: pd.DataFrame
    aspects: pd.DataFrame
    rubrics: pd.DataFrame
    test_users: pd.DataFrame

    __user_reverse_mapping: np.ndarray = field(init=False)
    __organisation_reverse_mapping: np.ndarray = field(init=False)

    def __post_init__(self):
        user_forward_mapping, self.__user_reverse_mapping = create_remap(self.users['user_id'])
        org_forward_mapping, self.__organisation_reverse_mapping = create_remap(self.organisations['org_id'])

        # Map everywhere relevant to new indices
        self.users['user_id'] = self.users['user_id'].map(user_forward_mapping)
        self.organisations['org_id'] = self.organisations['org_id'].map(org_forward_mapping)
        self.reviews['user_id'] = self.reviews['user_id'].map(user_forward_mapping)
        self.reviews['org_id'] = self.reviews['org_id'].map(org_forward_mapping)
        self.test_users['user_id'] = self.test_users['user_id'].map(user_forward_mapping)

    def map_users_back(self, user_ids: pd.Series) -> pd.Series:
        """
        Map users back from integer index to original user_id
        """
        return user_ids.map(lambda user_id: self.__user_reverse_mapping[user_id])

    def map_organizations_back(self, org_lists: pd.Series) -> pd.Series:
        """
        Map organizations back from integer indices to original org_ids.
        org_lists: pd.Series of arrays/lists of org_idx
        Returns: pd.Series of arrays/lists of org_ids (original)
        """
        return org_lists.apply(
            lambda org_list: self.__organisation_reverse_mapping[np.array(org_list)]
        )


@cache
def load_data(data_directory_path: str) -> Data:
    return Data(**{
        file.removesuffix('.csv'): pd.read_csv(os.path.join(data_directory_path, file), low_memory=False)
        for file in os.listdir(data_directory_path)
        if file.endswith('.csv')
    })

