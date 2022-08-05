from __future__ import annotations

from gym import Space
from trading_gym.dataloaders.dataloader import Dataloader
from typing import Dict, List, Optional, Literal, Tuple
from datetime import timedelta, date
import datetime
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import IntEnum


TimeInterval = Literal["1m"]
YEAR_DIFF = timedelta(365)


class KlineColumsIdx(IntEnum):
    IGNORE = 11
    TIMESTAMP = 0

    @staticmethod
    def n_columns():
        return 11
    
    @staticmethod
    def all_without(ignored: List[int]) -> Tuple[int]:
        return tuple(filter(lambda e: e not in ignored, range(KlineColumsIdx.n_columns() + 1)))


@dataclass
class ItemInfo:
    name: str
    min_date: date
    max_date: date
    csv_files: Tuple[Path]


class BinanceKlinesLoader(Dataloader):

    _path: str
    _item_white_list: List[str]
    _time_interval: TimeInterval
    _min_time_period: timedelta
    _items: List[ItemInfo]
    _step_in_file: int
    _file_idx: int
    _n_files_per_item: int
    _merged_df_rows: int
    _merged_df: pd.DataFrame

    def __init__(
        self, path: str,
        item_white_list: Optional[List[str]] = None,
        time_interval: TimeInterval = "1m",
        min_time_period: timedelta = YEAR_DIFF * 2,
        ) -> BinanceKlinesLoader:
        self._path = path
        self._time_interval = time_interval
        self._min_time_period = min_time_period
        self._item_white_list = item_white_list

        items = self.collect_item_info()
        if self._item_white_list is None:
            # if not set enable all
            self._item_white_list = [item.name for item in items]
        
        items = self.collect_item_info()
        self._items = self.filter_items(items)

        # load dfs
        self._step = 0
        self._file_idx = 0
        self._step_in_file = 0
        self._n_files_per_item = len(self._items[0].csv_files)
        self.load_data_from_nth_files(self._file_idx)
    
    def load_data_from_nth_files(self, file_idx: int):
        usecols = [KlineColumsIdx.all_without([KlineColumsIdx.IGNORE])] + [KlineColumsIdx.all_without([KlineColumsIdx.IGNORE, KlineColumsIdx.TIMESTAMP])] * (len(self._items) - 1)
        dframes = tuple((pd.read_csv(item.csv_files[file_idx], usecols=usecol, header=None) for item, usecol in zip(self._items, usecols)))

        # quality check
        n_rows = dframes[0].shape[0]
        self._merged_df_rows = n_rows
        for i, df in enumerate(dframes):
            assert df.shape[0] == n_rows,f"csv files must have same number of rows but {self._items[i].name} in file {self._items[i].csv_files[file_idx]} has {df.shape[0]} rows {n_rows} required"

        self._merged_df = pd.concat(
            dframes,
            axis=1,
            ignore_index=True
            )
        
    
    def collect_item_info(self) -> List[ItemInfo]:
        accepted_items: List[ItemInfo] = []
        base_path = Path(self._path)
        for item_path in base_path.glob(f"*/{self._time_interval}"):
            item_name = item_path.parent.name
            min_date = date(datetime.MAXYEAR, 1, 1)
            max_date = date(datetime.MINYEAR, 1, 1)
            skip_n_left = len(item_name) + len(self._time_interval) + 2
            skip_n_right = 4
            monthly_csv_files: List[Tuple[Path, date]] = []
            for monthly_csv_history in item_path.glob(f"*/{item_name}-{self._time_interval}-*.csv"):
                year, month = monthly_csv_history.name[skip_n_left:-skip_n_right].split("-")
                file_date = date(int(year), int(month), 1)
                min_date = min(min_date, file_date)
                max_date = max(max_date, file_date)
                monthly_csv_files.append((monthly_csv_history, file_date))

            monthly_csv_files.sort(key=lambda x: x[1])
            item_info = ItemInfo(
                name=item_name,
                min_date=min_date,
                max_date=max_date,
                csv_files=tuple(map(lambda x: x[0], monthly_csv_files))
            )
            accepted_items.append(item_info)
        return accepted_items
    
    def filter_items(self, items: List[ItemInfo]) -> List[ItemInfo]:
        filtered = []
        white_list = set(self._item_white_list)
        start_date = date(datetime.MAXYEAR, 1, 1)
        end_date = date(datetime.MINYEAR, 1, 1)
        for item in items:
            if item.name not in white_list:
                # skip item not included in items list
                continue
            if len(item.csv_files) == 0:
                # skip empty item
                continue
            if item.max_date - item.min_date < self._min_time_period:
                # skip item which recorded history is too short
                continue
            filtered.append(item)

            start_date = min(start_date, item.min_date)
            end_date = max(end_date, item.max_date)

        if len(filtered) == 0:
            raise ValueError(
                "No items that meet the criteria can are available"
                )
        
        if end_date - start_date < self._min_time_period:
            raise ValueError(
                "With choosen items training period where data is available for all items "
                f"is only {end_date - start_date} which is smaller then required {self._min_time_period}."
                )
        return filtered
    
    def __iter__(self):
        self._step_in_file = 0
        self._file_idx = 0
        self.load_data_from_nth_files(self._file_idx)
        return self
    
    def __next__(self) -> Tuple[np.ndarray, bool]:
        is_done = False
        if self._step_in_file == self._merged_df_rows - 1:
            self._step_in_file = 0
            self._file_idx += 1
            if self._file_idx == self._n_files_per_item - 1:
                is_done = True
                self._step_in_file = 0
                self._file_idx = 0
            self.load_data_from_nth_files(self._file_idx)
        
        observation = self._merged_df.iloc[self._step_in_file].to_numpy(dtype=np.float32)
        return observation, is_done    

if __name__ == "__main__":
    dl = BinanceKlinesLoader(
        path="/home/szysad/Projects/binance-public-data/python/data/spot/monthly/klines",
        item_white_list=["BTCUSDT"]
    )

    it = iter(dl)
    print(next(it))
