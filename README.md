# snake

Deep Reinforcement Learning проект для обучения агента игре "Змейка" с использованием DQN (Deep Q-Network).

## Обучение

Запустите обучение:

```bash
python train.py
```

Модели сохраняются в директорию `./models/`, состояния в `./states/`, логи TensorBoard в `./runs/`.

### Настройка гиперпараметров

Вы можете настроить гиперпараметры через аргументы командной строки:

```bash
python train.py --gamma 0.99 \
                --frame-count 7008000 \
                --eps-decay-time 0.5 \
                --episode-depth 10000 \
                --batch-size 32 \
                --eps-start 1.0 \
                --eps-end 0.05 \
                --q-targ-update-freq 32000
```

Доступные параметры:
- `--gamma` - коэффициент дисконтирования (default: 0.99)
- `--frame-count` - общее количество кадров для обучения (default: 7008000)
- `--eps-decay-time` - доля времени для уменьшения epsilon (default: 0.5)
- `--episode-depth` - максимальное количество шагов в эпизоде (default: 10000)
- `--batch-size` - размер батча для обучения (default: 32)
- `--eps-start` - начальное значение epsilon (default: 1.0)
- `--eps-end` - конечное значение epsilon (default: 0.05)
- `--q-targ-update-freq` - частота обновления целевой сети (default: 32000)

Посмотреть все параметры:

```bash
python train.py --help
```

### Сравнение гиперпараметров

TensorBoard автоматически записывает все гиперпараметры и финальные метрики. Вы можете запустить несколько экспериментов с разными параметрами и сравнить результаты во вкладке **HPARAMS** в TensorBoard:

```bash
# Эксперимент 1: стандартные параметры
python train.py

# Эксперимент 2: больший batch size
python train.py --batch-size 64

# Эксперимент 3: более медленный decay epsilon
python train.py --eps-decay-time 0.7 --eps-end 0.1
```

Все запуски будут сохранены в `./runs/` с уникальными метками времени и доступны для сравнения в TensorBoard.

## Визуализация с TensorBoard

Проект поддерживает визуализацию всех метрик через TensorBoard.

### Запуск TensorBoard

```bash
./start_tensorboard.sh
```

Или вручную:

```bash
tensorboard --logdir=./runs --port=6006
```

Откройте браузер по адресу: http://localhost:6006

### Доступные графики

#### Hyperparameters (гиперпараметры):
- **Hyperparameters/All** - все гиперпараметры тренировки (текст)
- **Environment/Configuration** - конфигурация игрового окружения (текст)
- **HPARAMS** (вкладка) - таблица сравнения запусков с разными гиперпараметрами и финальными метриками

#### Training (во время обучения):
1. **Training/Loss** - средняя ошибка обучения по эпизодам
2. **Training/Epsilon** - значение epsilon (exploration rate)
3. **Training/ReplayMemorySize** - размер буфера повтора
4. **Performance/Speed_it_per_s** - скорость обучения (итераций/сек)

#### Episode (статистика по эпизодам):
1. **Episode/FoodCount** - количество съеденной еды за эпизод
2. **Episode/FrameCount** - длительность эпизода в кадрах
3. **Episode/FoodPerFrame** - эффективность агента (еда/кадр)

#### Evaluation (оценка моделей):
Запустите оценку моделей:

```bash
python stats.py
```

Это создаст графики:
1. **Evaluation/MeanFramesCount** - средняя длительность игр
2. **Evaluation/MeanFoodCount** - среднее количество еды
3. **Evaluation/MeanLoopsRatio** - процент зацикливаний
4. **Evaluation/MeanFoodPerFrame** - средняя эффективность

## Демонстрация

Запустите обученную модель:

```bash
python demonstration.py
```
