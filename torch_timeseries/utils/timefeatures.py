from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

pd.options.mode.chained_assignment = None


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5

# 以上是一系列时间特征，把时间特征限制了-0.5到0.5之间
# 该函数主要就是根据freq返回对应的时间特征实例
def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

# 调用该函数，有三个参数，dates存储
def time_features(dates: pd.DataFrame, timeenc=1, freq='h') -> np.ndarray:
    """
    encode time features based on data sample freqency
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0: 
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday, ]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    > 
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]): 
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    if timeenc == 0:
        # 提取时间特征
        dates['month'] = dates.date.astype(object).apply(lambda row: row.month)
        # 从dates中提取日的数组
        dates['day'] = dates.date.astype(object).apply(lambda row: row.day)
        # 从dates中提取周几的数组
        dates['weekday'] = dates.date.astype(object).apply(lambda row: row.weekday())
        # 从dates中提取小时的数组
        dates['hour'] = dates.date.astype(object).apply(lambda row: row.hour)
        # 先提取分钟的数组，然后根据15分钟进行分组
        dates['minute'] = dates.date.astype(object).apply(lambda row: row.minute)
        dates['minute'] = dates.minute.map(lambda x: x // 15)

        # # 从dates中提取月的数组
        # dates['month'] = dates.date.apply(lambda row: row.month, 1)
        # # 从dates中提取日的数组
        # dates['day'] = dates.date.apply(lambda row: row.day, 1)
        # # 从dates中提取周几的数组
        # dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)
        # # 从dates中提取小时的数组
        # dates['hour'] = dates.date.apply(lambda row: row.hour, 1)
        # # 先提取分钟的数组，然后根据15分钟进行分组
        # dates['minute'] = dates.date.apply(lambda row: row.minute, 1)
        # dates['minute'] = dates.minute.map(lambda x: x // 15)
        # 指定每种频率能够提取哪些频率数组
        freq_map = {
            'y': [], 'm': ['month'], 'w': ['month'], 'd': ['month', 'day', 'weekday'],
            'b': ['month', 'day', 'weekday'], 'h': ['month', 'day', 'weekday', 'hour'],
            't': ['month', 'day', 'weekday', 'hour', 'minute'],
        }
        # 当freq为h，则就返回['month', 'day', 'weekday', 'hour']提取的数组
        # 结果形式   month  day  weekday  hour  minute
        # 0         7    1        4     2       0
        # 1         7    1        4     3       0
        # 2         7    1        4     4       0
        # 3         7    1        4     5       0
        # 4         7    1        4     6       0

        return dates[freq_map[freq.lower()]].values
    # 为1时
    if timeenc == 1:
        dates = pd.to_datetime(dates.date.values)
        # 通过freq获得对应的时间特征对象，然后将dates限定范围，也是最后要返回不同时间特征的数据
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1, 0)

if __name__=="__main__":
    print(time_features_from_frequency_str('h'))
