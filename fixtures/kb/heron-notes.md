# Heron: Entity Graphs for Tool Routing

**Author:** F. Adeyemi (2024)
**Id:** HX-2024-007
**Claim:** Entity-conditioned routing halves tool-selection errors versus flat tool lists.

## Abstract Notes

Heron tackles the growing problem of tool selection in agents that expose large
tool registries. As the number of available procedures climbs, models increasingly
misroute a task to the wrong tool, either because the flat list of definitions is
confusing or because relevant candidates are buried. The authors argue that a
registry organized purely by name or by a short description is the wrong shape for
selection.

Their proposal is to condition routing on the entities present in the user's
request. The agent first extracts the salient entities — people, systems, and
places — and then uses those entities to filter and rank the candidate tool set.
Tools are pre-associated with the entity types they typically act upon, so a query
about a particular system naturally surfaces only the procedures that operate on
that kind of system.

## Evaluation

Experiments across several tool-using benchmarks show that entity-conditioned
routing halves tool-selection errors when compared with a flat list of tool
definitions presented to the model. The benefit is largest for agents with many
tools and for requests that mention entities with strong tool affinities.

## Key Takeaways

Heron is directly relevant to semantic tool retrieval. It reinforces the idea that
retrieval should be keyed on rich semantic signals rather than on names, and that
entities extracted from the current request can meaningfully narrow the search
space. For a small static toolset the effect is modest, but the conceptual
framework scales to hundreds of procedures.
