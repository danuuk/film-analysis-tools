# Spektrafilm processing-architecture comparison v1

Status: completed source-tree engineering review; no migration or runtime change

Date: 2026-07-31

Spektrafilm pin: `3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`

Plugin reference: `obs-art-plugin@897aad096bade802abc864a6821d90820b587c27`

Typed-engine reference: `film-emulation-engine@a858187787b68e0d3eb34caded381fb86edc3910`

## Decision

Spektrafilm has the stronger *photographic process topology*. It keeps camera optics, emulsion
exposure, negative development, enlarger exposure, print development, viewing glare, and scanning
as distinct stages, so an effect can influence every later material response that it physically
precedes.

Our implementation has the stronger *production and governance architecture*. The shipping plugin
has bounded, proven GPU mechanisms and native parity machinery; the typed engine makes signal
domains, material identities, controls, array contracts, provenance, taps, and promotion state
explicit. These properties make failures local and keep an experimental mechanism from silently
becoming a production claim.

The right future direction is not to replace one architecture with the other. It is to retain our
typed, evidence-gated execution model while giving selected mechanisms more honest photographic
positions. In particular:

- keep the present plugin and accepted A1 grain unchanged;
- distinguish **negative emulsion scatter/back-reflection** from **post-print optical glow**;
- investigate a pre-negative halation candidate without removing the current guarded O3 control;
- evaluate a constrained monotone curve authoring model, compiled to our own immutable LUTs;
- evaluate DIR adjacency and print preflash as isolated candidates;
- keep Spektrafilm as a pinned external oracle and design input, never as runtime authority.

## Scope and evidence

This review inspected the actual source graphs and equations at the three revisions above. It also
reused the current plugin execution audit, typed-engine validation records, and the completed grain
oracle in [`spektrafilm-negative-grain-comparison-v1.md`](spektrafilm-negative-grain-comparison-v1.md).
No natural-image preference claim is made here. A full-render visual A/B between the projects would
confound stage architecture with different profile data, calibration, input transforms, gamut
mapping, and output placement.

One bounded numerical probe was therefore limited to curve integrity. Spektrafilm's stored 500T
negative curves contain only three tiny descending sample steps in total. Its stored 2383 samples
contain many small descents, but the runtime print stage does not interpolate those samples
directly: it evaluates a fitted sum of three normal CDF sublayers. That fit has no descending steps,
with per-channel RMS error `0.000705–0.000880` density and maximum error
`0.002736–0.004588` density against the stored samples. The current typed-engine assets retain
local descents in both 5219 and 2383. Existing engine evidence already shows that bounded isotonic
repair removes the observed synthetic reversals with a visually negligible aggregate change on the
reviewed camera cohort. The result supports a representation experiment, not adoption of the
Spektrafilm fit.

The reviewed Spektrafilm code is GPLv3 and its profiles carry separate share-alike terms. This
report contains observations and independently stated recommendations, not copied implementation.
Any future implementation must preserve that boundary or obtain explicit licensing advice or
permission.

## The three execution graphs

### Spektrafilm

```text
input RGB
  -> input decoding / spectral upsampling to film-layer exposure
  -> camera diffusion filter
  -> camera lens blur
  -> in-emulsion scatter + back-reflection halation
  -> filming exposure normalization and log exposure
  -> negative characteristic curves
  -> DIR same-layer/interlayer correction and redevelopment
  -> negative-density grain
  -> spectral negative transmittance
  -> enlarger illuminant + CMY filtration + optional preflash
  -> enlarger diffusion
  -> print characteristic-curve model/morph
  -> spectral viewing conversion
  -> print glare in XYZ
  -> output gamut compression
  -> scanner blur and unsharp mask
  -> output encoding
```

The decisive property is causality. Halation changes negative exposure before the negative curve,
couplers, grain amplitude, and print. Enlarger diffusion changes print exposure before the print
curve. Glare and scanner response remain later viewing/device effects.

### Current typed engine candidate

```text
Sony S-Log3 / S-Gamut3.Cine
  -> scene XYZ boundary
  -> 5219 spectral layer exposure and analytical density
  -> optional negative-density grain candidate
  -> 2383 spectral print exposure and analytical density
  -> spectral projection and display-output LUT
  -> relative RGB to L100
  -> O1 print MTF
  -> O2 guarded stock-acutance proxy
  -> O3 guarded halation
  -> L100 to relative RGB
  -> A1 appearance grain, when selected
```

The spectral colour core is materially structured and strongly typed. The spatial operators are a
later appearance block. O3 derives a print-density engagement proxy from final display RGB, but it
does not operate on actual negative density and cannot feed the film/print characteristics.

### Shipping Windows/macOS plugin

```text
OBS input preparation
  -> legacy AL2 blend endpoint
  -> B0' 5219 negative curves
  -> printer timing
  -> 2383 print curves
  -> display placement + FSC1-C stabilization
  -> post-print Timing and Chroma Gain
  -> print MTF
  -> guarded stock acutance
  -> guarded halation + optional extension
  -> grain-amplitude reference snapshot
  -> compact bloom -> print glow -> broad veil
  -> appearance grain
  -> OBS output
```

Linux currently executes a reduced monolithic path without the chain-only acutance extension and
late optical passes. This platform difference is an important cost in any proposal that requires a
new spatial pass inside the pointwise negative/print core.

## Component-by-component assessment

| Component | Spektrafilm | Our current implementation | Stronger approach now | Future disposition |
|---|---|---|---|---|
| Process topology | Explicit camera, emulsion, enlarger, print, view, and scanner stages | Coherent colour core followed by modular appearance layers | Spektrafilm for physical causality; ours for governance | Keep our contracts, improve selected stage placement |
| Negative/print curves | Sampled negative plus fitted three-sublayer CDF print model and anchor-preserving morph | Dense plugin LUTs; typed-engine 1024-sample curves; placement and safety repairs outside the curves | Split: Spektrafilm for well-formed authoring, ours for frozen runtime assets and validation | Test our own constrained offline fit, then compile LUTs |
| Spectral colour | Broad stock library, spectral upsampling, illuminants, filters, dyes, print, and scan | Shipping plugin is simplified; typed engine has a rigorous but narrow 5219/2383 spectral core | Spektrafilm for breadth; typed engine for contracts/provenance | Expand only through typed compiled assets |
| Density and masking | Spectral base and dye densities; explicit coupler model | Analytical density basis and shared masking control; no active native DIR coupling | Spektrafilm mechanism is stronger | Isolated nonspatial DIR experiment first |
| Halation | Layer-exposure scatter and multi-bounce back reflection before development | Bright/edge/area-gated tinted blur after projection/display | Spektrafilm for physical placement; ours for product safety | Keep O3; add a separately named pre-negative candidate |
| MTF and acutance | Stage-specific camera/enlarger/scanner blur and unsharp controls | Evidence-derived print MTF plus bounded log-luma acutance and headroom guards | Ours for the current validated mechanism | Preserve; reconsider placement only with stage evidence |
| Late optics | Camera diffusion families, enlarger diffusion, scanner blur/unsharp, print glare | Compact bloom, print glow, broad veil, and guarded energy/headroom behaviour | Split; neither is a complete physical model | Fit a common energy-accounted PSF basis by device stage |
| Grain | Negative-density sublayers and independent fields before print | Accepted A1 image-domain fallback plus N1/N2 negative-density research | A1 is stronger as current product evidence; Spektrafilm is a valuable structural oracle | Continue the independent N2 programme; do not transplant |
| Gamut and output | Explicit input-locus compression and several output compressors | Explicit scene boundary in the engine; FSC1-C/hard output constraints in current paths | Spektrafilm has stronger mapper breadth; ours has stronger domain declaration | Compare standards-based mappers independently |
| Controls and invariants | Rich live photographic controls, but broad mutable parameter surface | Typed controls, derived-asset invalidation, immutable bundles, promotion gates | Ours | Retain |
| Determinism and parity | Research Python runtime; some stochastic contracts are unsuitable for video | Film-frame cadence, deterministic seeds, native parity and platform audits | Ours | Retain |
| Provenance and claims | Profile-level attribution, but mechanisms and priors are not uniformly identified | Source hashes, evidence class, uncertainty, runtime status, promotion restrictions | Ours | Retain |

## Tone and characteristic curves

Spektrafilm has three related curve mechanisms:

1. Negative development interpolates a sampled per-channel density curve. Its `gamma_factor`
   changes contrast by scaling the log-exposure axis.
2. A generic analytic toe/straight-line/shoulder function exists for profile construction.
3. Print development evaluates three fitted CDF sublayers per channel. Its seven-control morph
   changes sublayer widths and centres together, can distinguish fast and slow populations, can
   blend toward a skewed developer-exhaustion response, preserves each sublayer amplitude and total
   maximum density, and re-anchors density at zero log exposure.

Our shipping B0' path samples dense negative and print LUT textures. It then adds a common-density
tone crosswalk, display placement, black/shoulder safety, and FSC1-C neutral/chroma stabilization.
The typed engine keeps negative exposure, negative density, print exposure, print density,
projected XYZ, and display RGB as different domains, but currently executes sampled characteristic
arrays without a monotonicity invariant.

The Spektrafilm print representation is the stronger *authoring primitive*: positive amplitudes and
widths make the CDF sum well formed, and its morph preserves meaningful anchors. It is also easier
to expose a small number of stable controls without editing a LUT arbitrarily. Our approach is the
stronger *deployment primitive*: a compiled LUT is cheap, deterministic, inspectable, and can match
the exact accepted curve rather than forcing a particular analytic family.

The recommended hybrid is an offline constrained model that produces our LUT. Requirements should
be ours: monotonicity or explicitly justified local exceptions, fixed D-min/D-max, declared LAD or
midscale anchor, bounded slope/curvature, and an error report against source samples. Spektrafilm's
three-CDF parameter values and developer-exhaustion implementation should not be copied.

Two current plugin mechanisms are weaker architecturally than either curve model:

- `Negative Process` crossfades the film core against an unrelated legacy tone renderer; this is a
  product macro, not development strength.
- `Timing` owns both printer-light changes before print density and creative dye operations after
  FSC1-C.

The first useful cleanup is semantic: expose the film renderer at full strength, label the legacy
endpoint as legacy, and separate printer timing from post-print colour. No curve change is required
for that cleanup.

## Halation: material response or optical appearance

Spektrafilm converts RGB to three film-sensitive exposure layers, applies camera diffusion and lens
blur, then applies two distinct film-plane operations in micrometres:

```text
E_scatter = (1-s)E + s[(1-w) Gaussian(E) + w Exponential(E)]
E_halated = E_scatter + sum_k a_k Gaussian(sigma * sqrt(k), E_scatter)
```

The back-reflection weights decay across bounces. Optional per-channel renormalization divides by
the uniform-field gain, so a flat field remains approximately unchanged while edges redistribute
energy. The result then passes through negative curves, DIR correction, grain, 2383, and scan.
This is not a wavelength-resolved transport model—it operates after spectral integration into three
sensitive layers—but it is a coherent material-stage approximation.

Our O3 derives brightness, edge, area, and colour engagement from display-linear Rec.709 in an
absolute-luminance block. It applies small and large tinted Gaussian contributions, guards titles
and broad bright areas, preserves headroom, exposes useful taps, and has exact controlled-input
parity with the frozen plugin implementation. These are substantial production advantages.

Its physical weakness is position. The source has already passed the negative curve, printer
exposure, 2383 curve, projection, output LUT/clamp, MTF, and acutance. A display-density proxy can
decide where an effect appears, but it cannot make halation alter negative saturation, coupler
release, grain response, or print roll-off. Headroom protection can also suppress the effect in the
very highlights where a pre-curve exposure perturbation would have been compressed naturally.

Therefore the choice is not “move O3” versus “keep O3.” They answer different questions:

- **negative emulsion scatter/back-reflection** should be tested before negative development;
- **post-print bloom/glow/veil** belongs after print formation;
- the existing O3 should remain an unchanged fallback until the first mechanism is independently
  evidenced and the product guard behaviour is reproduced intentionally.

The first halation experiment should use the same authored PSF and amplitude in three arms:

```text
H0: no halation
H1: layer-exposure halation before the 5219 curve
H2: current post-display O3
```

Measure neutral DC gain, point/edge radial energy, channel hue versus source hue, response across
the 5219 toe/straight-line/shoulder, effect after 2383, clipping/headroom intervention, subtitles,
large bright fields, and coloured practicals. Only after this placement experiment should PSF
families or stock-specific parameters be fitted.

## Colour, density, and couplers

Spektrafilm's colour graph is more complete than the shipping plugin: it reconstructs spectra for
camera exposure, forms spectral negative density, transmits an authored enlarger illuminant through
that density, applies subtractive printer filtration and optional preflash, exposes print
sensitivities, then converts spectral print transmission to XYZ under a viewing illuminant.

The typed engine implements the same essential 5219-to-2383 material separation with stronger
runtime contracts. Material IDs travel with exposure and density; a 5219 density cannot be consumed
as 2383 data. White-balance changes that require a rebuilt input LUT cannot masquerade as uniforms.
The shipping plugin sacrifices much of this spectral breadth for a compact GPU path and then uses
FSC1-C to stabilize neutral and opponent colour.

Spektrafilm's DIR mechanism is a genuine missing capability in our active native model. It builds a
same-layer/interlayer donor-to-receiver inhibition matrix, derives a pre-coupler curve so neutral
ramps still reproduce the profile's final characteristic, blurs released inhibitor with Gaussian
and exponential components, subtracts the correction from log exposure, and redevelops. That is
technically stronger than our current identity native coupling or the typed engine's scalar masking
blend.

It is not ready to migrate as-is. The model is a one-pass approximation; the parameters are not a
5219 validation result in our evidence system; the current ordering means grain does not feed back
into couplers; and spatial DIR would require negative development, a blur pass, and redevelopment
inside a currently fused plugin colour stage. A useful progression is:

1. nonspatial same-layer and interlayer matrix, with exact neutral-curve preservation;
2. saturated-patch and exposure-trajectory capacity test;
3. only then, spatial diffusion and adjacency comparison against O2 acutance;
4. reject any configuration that double-counts the measured characteristic curves or the existing
   acutance layer.

## Spatial and optical architecture

Spektrafilm's strongest optical idea is not one particular kernel; it is stage ownership. Camera
diffusion and lens blur precede emulsion exposure, enlarger diffusion precedes print development,
and scanner blur/unsharp follows viewing conversion. Its newer diffusion-filter families use
energy-conserving mixtures of isotropic exponential scales with core, halo, and long bloom groups.
That is a useful fitting basis for smooth tails and avoids visible knees between separately authored
glow effects.

Our strongest spatial mechanisms are the 2383-informed print MTF and the guarded log-luma acutance
operator. They have explicit film-plane scaling, headroom behaviour, controlled-input tests, and
native parity. Spektrafilm's general blur/unsharp controls do not displace that evidence.

Our weakest optical property is that compact bloom, print glow, and broad veil are three adjacent
artistic passes with partially overlapping responsibilities. Spektrafilm's multi-scale PSF basis
suggests a future unification experiment, but its private/prior-derived defaults are not evidence
for our parameters. Conversely, Spektrafilm's print-glare mechanism is only a blurred random
lognormal illuminant veil in XYZ; it is cleanly placed but not image-derived. Our broad veil has a
more useful image relationship. Placement and mechanism quality should be judged separately.

## Grain

The completed grain comparison remains controlling. Spektrafilm correctly places grain in negative
density before print and exposes useful sublayer and independent-field behaviour. Its deterministic
fixed field, reconstruction drift, spatial footprint, and chromatic extreme are unsuitable for our
runtime. A1 remains the accepted image-domain fallback. N2 should continue to combine our accepted
field and cadence with independently implemented density/layer variance and an explicit covariance
sweep.

## Additional mechanisms worth considering

| Mechanism | Why it is useful | Initial difficulty | Caution |
|---|---|---:|---|
| Monotone sublayer curve authoring | Removes accidental reversals and gives meaningful curve controls | Low; offline only | Do not force a poor analytic fit |
| Print preflash | Simple, physically located control over toe/colour before print development | Low to medium | Must be spectral or at least printer-layer aware |
| Separate printer filtration and exposure normalization | Makes timing controls correspond to an enlarger | Medium | Avoid hidden auto-normalization during comparison |
| Pre-negative emulsion scatter/halation | Restores interaction with curves, colour, grain, and print | Medium in engine; high in native chain | Preserve subtitles, DC energy, and headroom behaviour |
| Nonspatial then spatial DIR couplers | Adds missing saturation and adjacency mechanism | Medium to high | High double-counting risk |
| Unified multi-scale optical PSF | Could replace overlapping bloom/glow/veil tails | Medium | Fit from evidence; do not import family defaults |
| Explicit print-view glare | Separates viewing flare from emulsion halation | Medium | Spektrafilm's random veil is not sufficient authority |
| Input-locus and output-gamut compression | Addresses known boundary/floor behaviour more smoothly | Medium | Prefer independently specified standards and blind review |
| Enlarger and scanner stage taps | Makes print/scanner attribution testable | Low in engine | Native implementation may still be fused |
| UV/IR camera filtration and illuminant controls | Useful for broader stock/camera studies | High for live plugin use | Likely derived-asset rather than uniform controls |

## Adaptation and migration difficulty

No Spektrafilm code should be migrated. The table describes independent mechanism development.

### Easy or low-risk

- adopt clearer stage names in documentation and future UI;
- split printer timing from post-print colour semantics;
- fit constrained curves offline and compile the result to existing LUT contracts;
- add print preflash as an isolated print-exposure candidate;
- add device-stage taps and invariant tests;
- compare a standards-based output mapper without changing the film core.

### Moderate

- add an engine-only pre-negative halation operation with identity/off control;
- use an energy-accounted multi-scale PSF for one optical family;
- add nonspatial DIR coupling with neutral-curve preservation;
- separate print-view glare from image-derived late optics.

### Hard

- place spatial halation or DIR inside the native negative/print chain while preserving Windows,
  macOS, and reduced Linux behaviour;
- split the fused plugin colour pass around enlarger diffusion or spatial DIR redevelopment;
- make dynamic spectral illuminants, UV/IR filters, or stock changes live without rebuilding assets;
- run nine sublayer grain fields or a broad spectral stock library at OBS frame rates;
- prove that a newly positioned mechanism does not double-count the source curve, MTF, acutance,
  output mapper, or existing look grade.

## Development path

### Phase 0 — freeze and clarify

- Keep shipping defaults, current plugin numerics, typed-engine accepted/candidate states, and A1
  unchanged.
- Pin this Spektrafilm revision and source hashes as comparison evidence.
- Treat `Negative Process` and `Film Visibility` as product macros, not material parameters.
- Name emulsion halation, print optics, viewing glare, and scanner response separately in design
  documents even before execution changes.

### Phase 1 — bounded engine experiments

Run four independent candidates, each with an exact off/null control:

1. pre-negative halation placement versus current O3;
2. sampled curves versus our own constrained monotone fit;
3. nonspatial DIR matrix versus identity coupling;
4. current output boundary versus an independently specified smooth gamut mapper.

Do not combine them. Each report must include effect magnitude and spread, stage taps, wedge and
colour-patch behaviour, and a natural-scene pathology review where justified.

### Phase 2 — interaction tests

Only candidates that survive isolation should be combined. The critical pairs are halation with
negative curves, DIR with acutance, preflash with printer timing, output mapping with FSC1-C, and
negative grain with halation/print propagation. Test whether each pair explains distinct evidence
or merely applies the same visual change twice.

### Phase 3 — typed promotion before plugin work

Promote a selected mechanism to a typed engine profile only after provenance, invariants, and human
review are complete. Preserve the old recipe as a runnable fallback. Native shader work begins only
after the operation and its taps are frozen enough for parity testing.

### Phase 4 — plugin integration

Prefer offline compilation and existing passes. Add a new native pass only when stage placement is
the mechanism under test and cannot be represented without it. Require Windows/macOS parity and an
explicit Linux disposition before describing the plugin architecture as common across platforms.

## Final conclusion

Spektrafilm is stronger wherever photographic stage causality and process breadth matter most:
halation placement, DIR couplers, enlarger/preflash semantics, device separation, and constrained
print-curve authoring. Our system is stronger wherever a mechanism must be reproducible, bounded,
provenanced, testable, performant, and safely promoted into an OBS product.

The most important correction is architectural: halation should not be treated as one movable
effect. A pre-negative emulsion mechanism and a post-print optical mechanism are different causes
and may both be valid. The next revision should test that distinction in the typed engine while the
current O3 and A1 paths remain untouched.

This preserves the earlier corrected judgement at a broader scale: we were right not to copy
Spektrafilm casually, but wrong to exclude it from our thinking. Its best role is a pinned external
process oracle that helps us ask better questions and identify gaps in our own stage model.
