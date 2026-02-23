"""
Task Configurations for HCI Study

This module contains the task definitions, criteria, and 5 samples for each task.
"""

TASKS = {
    "T1": {
        "name": "Targeted Literature Search",
        "objective": "Find and synthesize relevant papers on a specific topic.",
        "criteria": [
            "Find at least 3-5 highly relevant papers on the given topic.",
            "Add the chosen papers to your Knowledge Base.",
            "Review the selected papers.",
            "Navigate to the 'Summary' tab and submit a brief literature overview of your findings."
        ],
        "samples": [
            "Application of Large Language/Vision Models in Medical Image Diagnostics.",
            "The impact of climate change on coastal urban infrastructure and mitigation strategies.",
            "Reinforcement learning algorithms (e.g., PPO, SAC) applied to robotic manipulation.",
            "Energy-efficient consensus mechanisms in emerging blockchain networks.",
            "Microplastics pollution in marine ecosystems and its measurable effect on local aquatic life."
        ]
    },
    "T2": {
        "name": "Deep Understanding of a Topic",
        "objective": "Thoroughly analyze literature to extract deep insights, limitations, and keywords.",
        "criteria": [
            "Find 1-3 highly technical papers on the specific topic and add them to your Knowledge Base.",
            "Deeply analyze the methodologies and results presented.",
            "Identify the main research gaps or future directions.",
            "Navigate to the 'Summary' tab to submit the Research Gaps and at least 5 relevant Keywords."
        ],
        "samples": [
            "Vision Transformers (ViT): How their self-attention mechanisms compare to traditional CNNs in feature extraction.",
            "CRISPR-Cas9 gene editing: Current challenges with off-target mutation rates and proposed mitigation strategies.",
            "Quantum error correction: The role of Surface Codes in building scalable topological quantum computers.",
            "Universal Basic Income (UBI): Analyzing the macroeconomic implications and inflation effects in pilot programs.",
            "Solid-state battery technology: Current material challenges in improving lithium-ion conductivity through solid electrolytes."
        ]
    }
}

TOOL_TUTORIALS = {
    "Manual": [
        "**Search Online:** Enter keywords to query the Semantic Scholar database and discover real academic papers.",
        "**Papers (Knowledge Base):** View the papers you've collected. Use filters (by year, author, keyword) to narrow down your list.",
        "**Analytics/Keywords:** Visualize publication trends over time and see a word cloud of the most common keywords in your collection.",
        "**Summary Tab:** Use this area to submit your final findings (Summary, Gaps, Keywords) to complete the task."
    ],
    "AI": [
        "**Search Online:** Standard keyword search to build your initial knowledge base.",
        "**Paper Chat:** Select a specific paper and ask the AI direct questions about its methodology, results, or limitations.",
        "**AI Summary:** Select multiple papers to automatically generate literature overviews, methodology comparisons, or key findings.",
        "**Research Insights:** Run a thematic analysis across multiple papers or get suggestions for how to cite them together.",
        "**Deep Research:** Describe what you're looking for in natural language. The AI will extract keywords, search online, remove duplicates, and filter papers based on relevance automatically."
    ]
}
