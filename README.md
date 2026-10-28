# Sound Car Scanner

Модульная система акустической локализации спецтранспорта и звуковых событий на дороге
по многоканальному микрофонному массиву.

**Текущий этап — 1: виртуальный полигон и ядро.** Ядро, оконный менеджер, профайлер, стример файлов
и конвейер работают; все алгоритмы пока заглушки, которые соблюдают контракт.

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_simulation.py --mode synthetic --azimuth 60 --duration 3   # тестовый .wav + GT .csv
python scripts/precompute_lut.py --resolution 1.0                             # LUT задержек
python pipeline.py --wav data/raw/synthetic_az60.wav                          # прогон + отчёт профайлера
pytest -q                                                                     # контрактные тесты
```

## Поток данных

```
BaseAudioSource (FileStreamer / LiveAudioDriver)
   └─ чанки (channels, n) произвольной длины
RealTimeWindowManager — фреймы 40 мс, шаг 20 мс, окно Ханна
   └─ X_frame (channels, samples), float32, read-only
Pipeline.process_frame
   1. pre_processing  — очистка (последовательно, каждый получает выход предыдущего)
   2. detectors       — триггеры событий
   3. localizers      — DOA; блоки с "trigger" запускаются только при срабатывании детектора
   4. distance        — получают лучший азимут (макс. confidence ≥ порога)
   └─ FrameResult → data/logs/*.csv
```

Бюджет всей цепочки — **< 20 мс на фрейм** (`realtime_budget_ms`), время каждого блока пишет `core/profiler.py`.

## Архитектурный контракт

Все блоки наследуются от классов из [core/interfaces.py](core/interfaces.py) и получают в конструктор
`geometry: ArrayGeometry`, `fs: int` и `params` из конфига.

| Тип (`kind`)    | Базовый класс           | Метод                                                   | Возврат                          |
|-----------------|-------------------------|---------------------------------------------------------|----------------------------------|
| `pre_processor` | `BasePreProcessor`      | `process(X_frame, fs)`                                  | `np.ndarray` той же формы        |
| `detector`      | `BaseEventDetector`     | `detect(X_frame, fs)`                                   | `(triggered: bool, score ∈[0,1])` |
| `localizer`     | `BaseDOALocalizer`      | `execute_doa_localization(X_frame, fs)`                 | `(angle ∈[0,360], conf ∈[0,1])`  |
| `distance`      | `BaseDistanceEstimator` | `execute_distance_estimation(X_frame, fs, angle\|None)` | `(distance_m ≥0 или nan, conf ∈[0,1])` |

Правила:

- `X_frame` — `(channels, samples)`, порядок каналов = порядок `mic_positions` в `config/array_config.json`.
- `X_frame` доступен только для чтения. Если нужно его изменить, сделайте копию.
- Прерасчёты (LUT, steering-векторы, фильтры) делаются в `prepare()`, а не на каждом фрейме.
- Состояние между фреймами (оценка шума, история энергии) сбрасывается в `reset()`.
- Азимут: 0° — вдоль +X, растёт против часовой стрелки, 90° — вдоль +Y. Координаты в метрах.
- Задержки: `x_m(t) = s(t − τ_m)`, `τ_m` относительно центра массива (`core.geometry.far_field_delays`).
- Если результат нарушает контракт, конвейер бросает `ContractViolation` с именем блока
  (`strict_contract: false` — результат просто отбрасывается).

## Как добавить свой алгоритм

1. Создайте файл в нужном домене `modules/<домен>/`. Регистрировать его в `__init__.py` не нужно,
   модули подхватываются автоматически.
2. Напишите класс:

   ```python
   from core.interfaces import BaseDOALocalizer
   from core.registry import register

   @register("localizer", "my_algo")
   class MyAlgo(BaseDOALocalizer):
       def prepare(self):
           self.lut = ...          # self.geometry, self.fs, self.params доступны

       def execute_doa_localization(self, X_frame, fs):
           return angle_degrees, confidence_score
   ```

3. Добавьте `{"name": "my_algo", "enabled": true, "params": {...}}` в нужную секцию
   `config/pipeline_config.json`.
4. `pytest -q`: контрактные тесты сами найдут и проверят новый блок.

Чтобы замерить отдельный внутренний шаг, используйте декоратор `@profile_real_time(budget_ms=3.0)`
или контекст `with measure("my_algo:fft"):` из `core.profiler`.

## Структура и ответственные

```
config/               array_config.json (геометрия), pipeline_config.json (окна, пороги, состав блоков)
core/                 interfaces.py, registry.py, window_manager.py (Троицкий), profiler.py, geometry.py, config.py
drivers/              base.py, file_streamer.py (HIL), audio_driver.py [на будущее]
modules/
  pre_processing/     denoiser.py — спектральное вычитание (Линьков)
  sirens/             music_localizer.py (Акулов), gcc_phat_siren.py (Кузин), srp_phat_siren.py (Сагитов)
  hearing/            onset_detector.py, gcc_phat_wide.py (Косолапов), srp_phat_wide.py (Балуев)
  fusion/             distance_spl.py (Алехина), distance_tdoa.py (Артемьев)
pipeline.py           оркестратор + CLI
scripts/              run_simulation.py (Сафаров), precompute_lut.py
tests/                контрактные тесты, тесты оконного менеджера, smoke-тест конвейера
data/                 raw/, lut/, logs/: не коммитится
cv_tracker/           [на будущее] yolo_tracker.py (YOLO + BoT-SORT), camera_calibration.py (пиксели → азимут/угол места)
```

## Форматы данных

- **Симуляция** (`data/raw/<name>.wav` + `<name>_gt.csv`): float32 WAV, каналы как в `array_config`;
  GT-колонки `timestamp_s, azimuth_deg, distance_m, x_m, y_m`.
- **Лог конвейера** (`data/logs/*.csv`), длинный формат, одна строка на пару (фрейм, блок):
  `frame_index, timestamp_s, kind, block, value, confidence, processing_ms`.
  `timestamp_s` — центр фрейма, его можно сопоставлять с GT напрямую.
- **LUT** (`data/lut/*.npy`): `<array>_far_field_<res>deg.npy` — задержки `(n_angles, channels)` в секундах,
  `<array>_far_field_<res>deg_azimuths.npy` — сетка азимутов `(n_angles,)`.

## Открытые вопросы

- `array_config.json` сейчас заполнен плейсхолдером (UCA, 8 микрофонов, R = 5 см). Нужна реальная геометрия.
- TDOA-триангуляция между несколькими массивами: текущий контракт рассчитан на один массив.
  Когда появится второй, расширим `Pipeline` и `BaseDistanceEstimator` до словаря `{array_id: X_frame}`.
- `run_simulation.py --mode pyroadacoustics` пока заглушка (задача Сафарова).
