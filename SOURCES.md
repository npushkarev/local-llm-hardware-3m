# Источники

Все ссылки, на которых стоят расчёты в [README.md](README.md) и [polnaya-versiya.html](polnaya-versiya.html).
Собрано 18 августа 2026.

В самих документах эти же ссылки разнесены по местам, где используются. В полной версии
есть отдельный раздел «Откуда взято каждое число» — сводная таблица с пометками
«подтверждено / исправлено / не проверено» по каждой величине.

---

## Документы

| Что | Ссылка |
| --- | --- |
| Полная версия, 18 разделов | https://claude.ai/code/artifact/eafe2352-2e48-41f8-9936-ba8b94eba4f8 |
| Короткая версия под Confluence | https://claude.ai/code/artifact/39d73388-142c-4b5d-b935-a8ba398215a1 |

---

## Замеры, на которых стоят расчёты

| Источник | Что даёт |
| --- | --- |
| [Dendro Logic — конкурентность на одном DGX Spark](https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/) | опорная точка ёмкости по чатам: 32 одновременных запроса → 7,95 т/с каждому, 256 → 3,62; gpt-oss-120b одиночным потоком 33,5 т/с |
| [0xdfi — GLM-5.2 на 4 × DGX Spark](https://github.com/0xdfi/GLM-5.2-1M-4x-DGX-Spark) | 819 т/с чтение, 29–42 т/с генерация, NVFP4 + MTP-5, контекст до 249к |
| [Форум NVIDIA — GLM-5.2 на 2 × Spark через RPC](https://forums.developer.nvidia.com/t/academic-glm-5-2-on-2x-dgx-spark-gb10-nodes-crazy-1-bit-ud-iq1-s-rpc-llama-cpp-256k-context-8-tok-s/374523) | 213 т/с чтение — то же железо вчетверо хуже на llama.cpp; автор признаёт конфигурацию непригодной |
| [Playbook NVIDIA по связке двух Spark](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md) | 189,85 Гбит/с по двум портам QSFP через RoCE |
| [StorageReview — обзор кластера DGX Spark](https://www.storagereview.com/review/nvidia-dgx-spark-cluster-review-distributed-inference-on-dell-gigabyte-and-hp) | кластер даёт ёмкость, а не скорость: «межузловая ткань становится доминирующей статьёй издержек» |
| [EXO Labs — Spark вместе с Mac Studio](https://blog.exolabs.net/nvidia-dgx-spark/) | Spark читает контекст в 3,8 раза быстрее M3 Ultra, M3 Ultra генерирует в 3,4 раза быстрее Spark; ≈100 против ≈26 TFLOPS FP16 |
| [llm-tracker — Strix Halo](https://llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max+-395)-GPU-Performance) | 215 ГБ/с на практике против заявленных 256; наш замер — 212 |
| [vLLM на DGX Spark](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark) | пара 1 884 / 22,7–23,7 т/с, на которой строились первоначальные расчёты; prefill измерен только до 7,2к |
| [NVIDIA — gpt-oss на Spark](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/) | сравнение чтения контекста на Spark и Strix Halo |

Собственные замеры — 35 прогонов `llama-bench` и 24 прогона качества на личном стенде
(два DGX Spark и мини-ПК Strix Halo с RTX 4070 Ti по OCuLink) — внешнего подтверждения
не имеют и помечены в документах как требующие повторного прогона на свободных узлах.

---

## Железо

| Позиция | Ссылка |
| --- | --- |
| DGX Spark: 128 ГБ, 273 ГБ/с | [docs.nvidia.com](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) · [Chips and Cheese](https://chipsandcheese.com/p/analyzing-nvidia-gb10s-gpu) |
| Подорожание Spark до $4 699 | [объявление NVIDIA, 23.02.2026](https://forums.developer.nvidia.com/t/2-23-2026-price-change-announcement/361713) |
| RTX PRO 6000 Blackwell: 96 ГБ, 1792 ГБ/с, 600 Вт | [NVIDIA](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/) · [PNY](https://www.pny.com/nvidia-rtx-pro-6000-blackwell-ws) |
| RTX PRO 6000 Max-Q: 300 Вт | [datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/workstation-datasheet-blackwell-rtx-pro-6000-max-q-nvidia-3519233.pdf) |
| RTX PRO 5000 Blackwell 72 ГБ | [NVIDIA](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/) |
| RTX 4090 48 ГБ, разбор модификации | [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/blower-style-rtx-4090-48gb-teardown-reveals-dual-sided-memory-configuration-pcb-design-echoes-the-rtx-3090) · [VideoCardz](https://videocardz.com/newz/custom-geforce-rtx-4090-48gb-now-comes-with-water-cooling-sales-of-modded-48gb-cards-booming-in-china) |
| H200: 141 ГБ, 4,8 ТБ/с | [Spheron](https://www.spheron.network/blog/nvidia-h200-specs/) · [TRG](https://www.trgdatacenters.com/resource/nvidia-h200-vs-h100/) |
| H100: 80 ГБ, 3,35 ТБ/с | [HorizonIQ](https://www.horizoniq.com/blog/h200-vs-h100/) |
| AMD Ryzen AI Max+ 395 (Strix Halo) | [AMD](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html) |
| EPYC 9005: 12 каналов, 576 ГБ/с заявлено | [datasheet AMD](https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/datasheets/amd-epyc-9005-series-processor-datasheet.pdf) · [Phoronix: STREAM TRIAD 348 ГБ/с](https://www.phoronix.com/forums/forum/hardware/processors-memory/1507066-8-vs-12-channel-ddr5-6000-memory-performance-with-amd-5th-gen-epyc) |
| Apple Mac Studio M3 Ultra | [Apple Newsroom](https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/) |
| ROCm 7.1 для Ryzen | [AMD docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html) |

---

## Модели и бенчмарки

| Модель | Ссылка |
| --- | --- |
| Laguna S 2.1 — 118B/8B, 1M контекста, TB 2.1 = 70,2 | [Hugging Face](https://huggingface.co/poolside/Laguna-S-2.1) · [MarkTechPost](https://www.marktechpost.com/2026/07/21/poolside-releases-laguna-s-2-1/) |
| DeepSeek-V4-Flash — 284B/13B, ≈167 ГБ файлы, 148,7 ГиБ в памяти | [рецепт vLLM](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) · [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 — 743–753B/~40B, MIT, 1M контекста | [Artificial Analysis](https://artificialanalysis.ai/models/glm-5-2) · [morphllm](https://www.morphllm.com/glm-5-2) |
| GLM-5.3 — та же база, прирост от пост-тренинга | [explainx](https://explainx.ai/blog/glm-5-3-launch-cyber-defense-benchmarks-august-2026) · [Wccftech](https://wccftech.com/zhipus-glm-5-3-matches-fable-5-on-coding-using-only-post-training-and-stuns-fans-by-unearthing-a-vulnerability-all-the-way-from-1981/) |
| Qwen3.8-27B — 27,78B плотная, 262к контекста, Apache 2.0 | [BenchLM](https://benchlm.ai/models/qwen3-8-27b) · [Yotta Labs](https://www.yottalabs.ai/post/qwen-3-8-27b-specs-hardware-requirements-how-to-run-2026) |
| Terminal-Bench 3.0 — GLM-5.2 = 4,6% при лидере 42,7% | [Snorkel AI](https://snorkel.ai/leaderboard/terminal-bench-3-0/) · [tbench.ai](https://www.tbench.ai/) |
| Кэш префикса в vLLM | [docs.vllm.ai](https://docs.vllm.ai/en/latest/design/prefix_caching/) |
| Тарифы DeepSeek API | [api-docs.deepseek.com](https://api-docs.deepseek.com/quick_start/pricing) |

---

## Цены

### Розница, 18 августа 2026

| Продавец | Цена DGX Spark |
| --- | --- |
| [NIX.ru](https://www.nix.ru/autocatalog/other_computers/NVIDIA-DGX-Spark-4TB-940-54242-0006-000-ARM-v92-A-GB10-128-4TbSSD-NVIDIA-Grace-Blackwell-WiFi-BT-DGX-OS_960944.html) | 583 588 ₽ |
| [ИНФОТЕХ](https://i-teh.com/catalog/sistemnyy_blok/nvidia_dgx_spark_940_54242_0006_000_superkompyuter/) | 696 902 ₽ |
| [TEHPOS](https://tehpos.ru/elektrostal/nvidia-940-54242-0006-000.html) | 714 027 ₽ |
| [ONPAD](https://onpad.ru/catalog/cubie/nvidia/3964.html) | 726 900 ₽ |
| [Регард](https://www.regard.ru/product/763551/superkompiuter-nvidia-dgx-spark-940-54242-0006-000) · [Kvantech](https://kvan.tech/catalog/servery-i-vychislitelnye-sistemy/server-nvidia-dgx-spark/) · [OZON](https://www.ozon.ru/product/kompaktnyy-superkompyuter-nvidia-dgx-spark-s-ii-128gb-lpddr5-4tb-ssd-3377377344/) | цена уточняется у продавца |

Рекомендованная цена NVIDIA — $4 699, то есть 397 253 ₽ по курсу 84,54. Наценка розницы — от 1,47 до 1,83 раза.

### Вторичный рынок

Ссылки ведут на **поисковые запросы, а не на конкретные объявления**: объявления живут недели,
запрос воспроизводим в любой момент. Цены в объявлениях указаны за наличный расчёт физлицу —
к ним применимы наценки от 7 до 22% при оплате по счёту с НДС.

| Позиция | Запрос | Типично, наличными |
| --- | --- | --- |
| DGX Spark 128 ГБ / 4 ТБ | [avito.ru](https://www.avito.ru/all?q=DGX+Spark) | 450 000 – 500 000 ₽ |
| RTX PRO 6000 Blackwell WE 96 ГБ | [avito.ru](https://www.avito.ru/all?q=RTX+PRO+6000+Blackwell) | 999 000 – 1 100 000 ₽ |
| RTX PRO 6000 Max-Q 96 ГБ | [avito.ru](https://www.avito.ru/all?q=RTX+PRO+6000+Max-Q) | 1 428 000 – 1 690 000 ₽ |
| RTX PRO 6000 Server Edition | [avito.ru](https://www.avito.ru/all?q=RTX+PRO+6000+Server+Edition) | 1 550 000 ₽ |
| RTX PRO 5000 Blackwell 72 ГБ | [avito.ru](https://www.avito.ru/all?q=RTX+PRO+5000+Blackwell) | 979 800 ₽ |
| RTX 4090 48 ГБ доработанная | [avito.ru](https://www.avito.ru/all?q=RTX+4090+48GB) | 370 000 – 382 000 ₽ |
| Мини-ПК Strix Halo 128 ГБ | [avito.ru](https://www.avito.ru/all?q=Ryzen+AI+Max+395) | 229 000 ₽ |
| H200 | [avito.ru](https://www.avito.ru/all?q=NVIDIA+H200) | 3 999 000 ₽ |

**Ни одна цена не является офертой и не заменяет коммерческое предложение.**
