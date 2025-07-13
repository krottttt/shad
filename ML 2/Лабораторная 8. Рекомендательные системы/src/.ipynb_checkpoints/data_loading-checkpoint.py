import os
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import pandas as pd


def create_remap(series: pd.Series) -> tuple[dict, np.ndarray]:
    """
    return two mappers from ids to the smallest integer set and back (one dict and one ndarray for speed)
    :param series: series with ids
    :return:
    """
    assert series.is_unique, "Series must have unique values"
    raise NotImplementedError('TASK')


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

        self.users['user_id'] = self.users['user_id'].map(user_forward_mapping.get).values
        self.organisations['org_id'] = self.organisations['org_id'].map(org_forward_mapping.get).values
        self.reviews['user_id'] = self.reviews['user_id'].map(user_forward_mapping.get).values
        self.reviews['org_id'] = self.reviews['org_id'].map(org_forward_mapping.get).values
        self.test_users['user_id'] = self.test_users['user_id'].map(user_forward_mapping.get).values

    def map_users_back(self, user_ids: pd.Series) -> pd.Series:
        """
        map users back :)
        :param user_ids:
        :return:
        """
        raise NotImplementedError('TASK')
        return pd.Series(new_values, name=user_ids.name, index=user_ids.index)

    def map_organizations_back(self, org_lists: pd.Series) -> pd.Series:
        """
        map organizations back :(
        :param org_lists: pd.Series containing uneven np.ndarrays of org ids (in integer representation)
        :return: pd.Series containing uneven np.ndarrays of org ids (in original meaning)
        """
        try:
            raise NotImplementedError('TASK')
            return pd.Series(new_values, name=org_lists.name, index=org_lists.index)
        except ValueError as _:
            return org_lists.apply(lambda org_list: ...)


@cache
def load_data(data_directory_path: str) -> Data:
    return Data(**{
        file.removesuffix('.csv'): pd.read_csv(os.path.join(data_directory_path, file), low_memory=False) for file in
        os.listdir(data_directory_path)
    })
