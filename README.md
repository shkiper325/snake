# Snake DQN

Deep Reinforcement Learning проект для обучения агента игре "Змейка" с использованием DQN (Deep Q-Network).

## Возможности

- ✅ **DQN обучение** с replay memory и target network
- ✅ **TensorBoard интеграция** для визуализации всех метрик
- ✅ **Headless режим** для серверов без дисплея
- ✅ **Настраиваемые гиперпараметры** через CLI
- ✅ **AlterNet** - загрузка альтернативной модели для transfer learning
- ✅ **MSE регуляризация** между весами Q-сети и AlterNet
- ✅ **Организация экспериментов** с раздельным хранением моделей
- ✅ **Красивый терминальный вывод** с progress bar и ETA

## Быстрый старт

### Базовое обучение

```bash
python train.py
```

Модели сохраняются в `./models/`, состояния в `./states/`, TensorBoard логи в `./runs/`.

### Визуализация

```bash
./start_tensorboard.sh
```

Откройте http://localhost:6006 в браузере для просмотра метрик.

### Демонстрация

```bash
python demonstration.py
```

## Полное руководство

### 1. Настройка гиперпараметров

Все гиперпараметры настраиваются через аргументы командной строки:

```bash
python train.py \
    --gamma 0.99 \
    --frame-count 7008000 \
    --eps-decay-time 0.5 \
    --episode-depth 10000 \
    --batch-size 32 \
    --eps-start 1.0 \
    --eps-end 0.05 \
    --q-targ-update-freq 32000
```

#### Доступные параметры

**Основные параметры обучения:**
- `--gamma` - коэффициент дисконтирования (default: 0.99)
- `--frame-count` - общее количество кадров для обучения (default: 7008000)
- `--batch-size` - размер батча для обучения (default: 32)

**Exploration (epsilon-greedy):**
- `--eps-start` - начальное значение epsilon (default: 1.0)
- `--eps-end` - конечное значение epsilon (default: 0.05)
- `--eps-decay-time` - доля времени для уменьшения epsilon (default: 0.5)

**Параметры сети:**
- `--q-targ-update-freq` - частота обновления целевой сети в кадрах (default: 32000)
- `--episode-depth` - максимальное количество шагов в эпизоде (default: 10000)

**Управление:**
- `--out-dir` - директория для сохранения моделей и состояний (default: текущая)
- `--headless` - запуск без визуализации для серверов без дисплея

**AlterNet и регуляризация:**
- `--alter-net` - путь к файлу модели для загрузки как AlterNet
- `--alter-loss-lambda` - коэффициент MSE регуляризации (default: 1.0)

Полный список параметров:

```bash
python train.py --help
```

### 2. Headless режим (для серверов)

Для запуска на сервере без дисплея:

```bash
# Через флаг
python train.py --headless

# Через переменную окружения
SNAKE_HEADLESS=1 python train.py

# С другими параметрами
python train.py --headless --batch-size 64 --frame-count 5000000
```

**Особенности headless режима:**
- Pygame не инициализируется (не требуется дисплей/X-сервер)
- Все вычисления происходят без визуализации
- TensorBoard логирование работает полностью
- Можно запускать на серверах через SSH

### 3. Запуск на сервере в фоновом режиме

#### Готовый скрипт

```bash
./train_server.sh
```

Скрипт запускает два параллельных эксперимента в фоне.

#### Ручной запуск с nohup

```bash
nohup python train.py --headless --out-dir experiment1 > train.log 2>&1 &

# Мониторинг прогресса
tail -f train.log

# Просмотр процессов
ps aux | grep train.py

# Остановка обучения
kill <PID>
```

#### Просмотр TensorBoard через SSH

На локальной машине создайте туннель:

```bash
ssh -L 6006:localhost:6006 user@server
```

На сервере запустите TensorBoard:

```bash
./start_tensorboard.sh
```

Откройте http://localhost:6006 в браузере на локальной машине.

### 4. AlterNet - альтернативная сеть

AlterNet позволяет загрузить предобученную модель для:
- Transfer learning от другой модели
- Регуляризации обучения через MSE loss
- Сравнения производительности разных моделей

#### Базовое использование

```bash
# Загрузка модели как AlterNet
python train.py --alter-net ./experiment1/models/1000000

# С указанием выходной директории
python train.py \
    --out-dir new_experiment \
    --alter-net ./old_experiment/models/5000000
```

#### MSE регуляризация

При загрузке AlterNet к loss функции добавляется регуляризационное слагаемое:

```
total_loss = dqn_loss + lambda * MSE(Q_weights, AlterNet_weights)
```

MSE вычисляется между весами и bias'ами обоих линейных слоев (lin1 и lin2).

**Настройка коэффициента регуляризации:**

```bash
# По умолчанию lambda = 1.0
python train.py --alter-net ./models/checkpoint

# Уменьшить влияние регуляризации
python train.py --alter-net ./models/checkpoint --alter-loss-lambda 0.5

# Отключить регуляризацию (использовать AlterNet только для сравнения)
python train.py --alter-net ./models/checkpoint --alter-loss-lambda 0.0

# Усилить регуляризацию
python train.py --alter-net ./models/checkpoint --alter-loss-lambda 2.0
```

**Примечание:** AlterNet всегда загружается в режиме `eval()` и не обновляется во время обучения.

### 5. Организация экспериментов

Используйте `--out-dir` для раздельного хранения моделей:

```bash
# Эксперимент 1: стандартные параметры
python train.py --out-dir experiment1

# Эксперимент 2: больший batch size
python train.py --out-dir experiment2 --batch-size 64

# Эксперимент 3: медленный epsilon decay
python train.py --out-dir experiment3 --eps-decay-time 0.7 --eps-end 0.1

# Эксперимент 4: с AlterNet регуляризацией
python train.py \
    --out-dir experiment4 \
    --alter-net ./experiment1/models/3000000 \
    --alter-loss-lambda 0.5
```

#### Структура директорий

```
.
├── experiment1/
│   ├── models/           # Сохраненные веса моделей
│   │   ├── 70080         # Чекпоинт на 70080 кадре
│   │   ├── 140160
│   │   └── ...
│   └── states/           # Состояния для продолжения обучения
│       ├── 70080
│       └── ...
├── experiment2/
│   ├── models/
│   └── states/
└── runs/                 # Общие TensorBoard логи
    ├── Dec31_10-30-45_hostname/
    ├── Dec31_14-22-33_hostname/
    └── ...
```

### 6. Сравнение экспериментов

TensorBoard автоматически логирует все гиперпараметры и метрики.

Во вкладке **HPARAMS** в TensorBoard вы можете:
- Сравнить метрики разных запусков
- Отсортировать по финальной производительности
- Проанализировать влияние гиперпараметров
- Построить scatter plots и parallel coordinates

## TensorBoard визуализация

### Запуск

```bash
./start_tensorboard.sh
# или
tensorboard --logdir=./runs --port=6006
```

Откройте http://localhost:6006

### Доступные графики

#### Hyperparameters
- **Hyperparameters/All** - все гиперпараметры тренировки (текстовая таблица)
- **Environment/Configuration** - конфигурация игрового окружения
- **AlterNet/Configuration** - информация о загруженной AlterNet (если есть)
- **HPARAMS** (вкладка) - интерактивная таблица сравнения запусков

#### Training (метрики обучения)
- **Training/Loss** - средний loss по эпизодам
- **Training/DQN_Loss** - компонента DQN loss (если используется AlterNet)
- **Training/AlterNet_Regularization** - компонента MSE регуляризации (если используется AlterNet)
- **Training/Epsilon** - текущее значение epsilon (exploration rate)
- **Training/ReplayMemorySize** - размер replay buffer

#### Episode (статистика по эпизодам)
- **Episode/FoodCount** - количество съеденной еды за эпизод
- **Episode/FrameCount** - длительность эпизода в кадрах
- **Episode/FoodPerFrame** - эффективность агента (еда/кадр)

#### Performance
- **Performance/Speed_it_per_s** - скорость обучения (итераций в секунду)

#### Evaluation (оценка моделей)

Для получения evaluation метрик запустите:

```bash
# Обычный режим
python stats.py --models-dir ./experiment1/models

# Headless режим
python stats.py --headless --models-dir ./experiment1/models

# С настройкой количества игр
python stats.py --headless --models-dir ./experiment1/models --games 1000
```

Это создаст графики:
- **Evaluation/MeanFramesCount** - средняя длительность игр
- **Evaluation/MeanFoodCount** - среднее количество съеденной еды
- **Evaluation/MeanLoopsRatio** - процент зацикливаний
- **Evaluation/MeanFoodPerFrame** - средняя эффективность

## Примеры использования

### Простое обучение

```bash
python train.py
```

### Длительное обучение с настройками

```bash
python train.py \
    --out-dir long_training \
    --frame-count 10000000 \
    --batch-size 64 \
    --eps-decay-time 0.6
```

### Transfer learning с регуляризацией

```bash
# Сначала обучите базовую модель
python train.py --out-dir base_model --frame-count 3000000

# Затем обучите новую модель с регуляризацией от базовой
python train.py \
    --out-dir transfer_model \
    --alter-net ./base_model/models/3000000 \
    --alter-loss-lambda 0.8 \
    --frame-count 5000000
```

### Параллельное обучение на сервере

```bash
# Запуск нескольких экспериментов параллельно
SNAKE_HEADLESS=1 nohup python train.py --out-dir exp1 --batch-size 32 > exp1.log 2>&1 &
SNAKE_HEADLESS=1 nohup python train.py --out-dir exp2 --batch-size 64 > exp2.log 2>&1 &
SNAKE_HEADLESS=1 nohup python train.py --out-dir exp3 --batch-size 128 > exp3.log 2>&1 &

# Мониторинг
tail -f exp1.log
```

### Эксперимент с разными lambda

```bash
# Загрузить базовую модель
BASE_MODEL=./base/models/2000000

# Попробовать разные коэффициенты регуляризации
python train.py --out-dir lambda_0.0 --alter-net $BASE_MODEL --alter-loss-lambda 0.0
python train.py --out-dir lambda_0.5 --alter-net $BASE_MODEL --alter-loss-lambda 0.5
python train.py --out-dir lambda_1.0 --alter-net $BASE_MODEL --alter-loss-lambda 1.0
python train.py --out-dir lambda_2.0 --alter-net $BASE_MODEL --alter-loss-lambda 2.0
```

## Оценка моделей

### Базовая оценка

```bash
python stats.py --models-dir ./models
```

### Headless оценка с большим количеством игр

```bash
python stats.py \
    --headless \
    --models-dir ./experiment1/models \
    --games 1000
```

Результаты автоматически логируются в TensorBoard в директорию `./runs/evaluation`.

## Структура проекта

```
snake/
├── train.py              # Основной скрипт обучения
├── stats.py              # Оценка обученных моделей
├── demonstration.py      # Демонстрация игры
├── QNet.py              # Архитектура нейронной сети
├── env.py               # Игровое окружение
├── replay_memory.py     # Replay buffer
├── utils.py             # Вспомогательные функции
├── use_cuda.py          # Определение устройства (CPU/CUDA)
├── start_tensorboard.sh # Скрипт запуска TensorBoard
├── train_server.sh      # Скрипт для обучения на сервере
├── requirements.txt     # Зависимости (с pygame)
├── requirements-headless.txt  # Зависимости для серверов
└── README.md           # Документация
```

## Установка зависимостей

### Для локальной машины (с визуализацией)

```bash
pip install -r requirements.txt
```

### Для сервера (headless)

```bash
pip install -r requirements-headless.txt
```

## Технические детали

### Архитектура сети

```python
QNet:
  Conv2d(k → 16, kernel=3x3)
  ReLU
  Conv2d(16 → 32, kernel=3x3)
  ReLU
  Flatten
  Linear(1152 → 256)
  ReLU
  Linear(256 → 4)  # 4 действия: вверх, вниз, влево, вправо
```

### Алгоритм

- **Алгоритм:** Deep Q-Network (DQN)
- **Replay memory:** Prioritized experience replay (500,000 переходов)
- **Target network:** Обновляется каждые 32,000 кадров
- **Loss function:** SmoothL1Loss (Huber loss)
- **Optimizer:** Adam (lr=0.0000625, eps=1.5e-4)
- **Exploration:** Epsilon-greedy с экспоненциальным decay
- **MSE регуляризация:** exp(-MSE) между весами Q и AlterNet (опционально)

### Игровая среда

- **Поле:** 8x8 клеток
- **Начальная длина змейки:** 4
- **Награды:** еда=+1, смерть=-1, выживание=0
- **Наблюдение:** 4 последних кадра (grayscale, нормализованные z-score)

## FAQ

**Q: Как продолжить обучение после остановки?**

A: Просто запустите скрипт с теми же параметрами - обучение продолжится с последнего чекпоинта:

```bash
python train.py --out-dir experiment1
```

**Q: Как выбрать оптимальный batch size?**

A: Зависит от доступной памяти GPU/CPU. Рекомендуемые значения: 32, 64, 128.

**Q: Когда использовать AlterNet регуляризацию?**

A: Когда у вас есть предобученная модель и вы хотите:
- Ограничить отклонение новой модели от базовой
- Стабилизировать обучение
- Реализовать transfer learning

**Q: Что делать, если lambda слишком большая?**

A: Модель будет слишком сильно привязана к AlterNet и не сможет обучаться. Уменьшите lambda до 0.1-0.5.

**Q: Как интерпретировать графики в TensorBoard?**

A:
- `FoodCount` растет → агент учится
- `Loss` снижается → сеть сходится
- `Epsilon` падает → меньше exploration, больше exploitation
- `FoodPerFrame` растет → агент становится эффективнее

**Q: Можно ли использовать несколько GPU?**

A: Текущая реализация использует одну GPU. Для multi-GPU нужна доработка.

## Лицензия

MIT

## Контакты

Вопросы и предложения приветствуются в Issues.
