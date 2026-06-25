// nanobind wrapper around Clipper2 (https://github.com/AngusJohnson/Clipper2).
//
// Two operations are exposed, both working purely in the XY plane on
// double-precision coordinates:
//
//   offset_paths  -- Clipper2's InflatePaths (polygon / open-path offsetting)
//   clip_lines    -- intersect OPEN polylines with closed boundary polygons,
//                    i.e. keep only the parts of the lines that fall inside the
//                    boundary.  This is what the Python `hatch` helper builds on.
//
// Geometry crosses the language boundary as plain Python lists: a path is a
// list of [x, y] pairs, and a set of paths is a list of those.  Clipper2 has no
// other dependencies, so there is deliberately no Eigen / numpy buffer protocol
// here -- the data volumes for CNC contours and hatches are small and the
// list-of-lists conversion keeps the binding trivial to read.

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "clipper2/clipper.h"

namespace nb = nanobind;
using namespace nb::literals;

// Python-facing representation: each point is [x, y].
using Pt2 = std::array<double, 2>;
using Path2 = std::vector<Pt2>;
using Paths2 = std::vector<Path2>;

using Clipper2Lib::ClipperD;
using Clipper2Lib::ClipType;
using Clipper2Lib::EndType;
using Clipper2Lib::FillRule;
using Clipper2Lib::JoinType;
using Clipper2Lib::PathD;
using Clipper2Lib::PathsD;
using Clipper2Lib::PointD;

// --------------------------------------------------------------------------
// conversions
// --------------------------------------------------------------------------

static PathsD to_clipper(const Paths2 &paths)
{
    PathsD out;
    out.reserve(paths.size());
    for (const auto &path : paths)
    {
        PathD cpath;
        cpath.reserve(path.size());
        for (const auto &pt : path)
            cpath.emplace_back(pt[0], pt[1]);
        out.push_back(std::move(cpath));
    }
    return out;
}

static Paths2 from_clipper(const PathsD &paths)
{
    Paths2 out;
    out.reserve(paths.size());
    for (const auto &cpath : paths)
    {
        Path2 path;
        path.reserve(cpath.size());
        for (const auto &pt : cpath)
            path.push_back(Pt2{pt.x, pt.y});
        out.push_back(std::move(path));
    }
    return out;
}

static JoinType to_join_type(int value)
{
    switch (value)
    {
    case 0: return JoinType::Square;
    case 1: return JoinType::Bevel;
    case 2: return JoinType::Round;
    case 3: return JoinType::Miter;
    default: throw std::invalid_argument("join_type must be 0=Square, 1=Bevel, 2=Round, 3=Miter");
    }
}

static EndType to_end_type(int value)
{
    switch (value)
    {
    case 0: return EndType::Polygon;
    case 1: return EndType::Joined;
    case 2: return EndType::Butt;
    case 3: return EndType::Square;
    case 4: return EndType::Round;
    default: throw std::invalid_argument("end_type must be 0=Polygon, 1=Joined, 2=Butt, 3=Square, 4=Round");
    }
}

static FillRule to_fill_rule(int value)
{
    switch (value)
    {
    case 0: return FillRule::EvenOdd;
    case 1: return FillRule::NonZero;
    case 2: return FillRule::Positive;
    case 3: return FillRule::Negative;
    default: throw std::invalid_argument("fill_rule must be 0=EvenOdd, 1=NonZero, 2=Positive, 3=Negative");
    }
}

// --------------------------------------------------------------------------
// operations
// --------------------------------------------------------------------------

static Paths2 offset_paths(
    const Paths2 &paths,
    double delta,
    int join_type,
    int end_type,
    double miter_limit,
    double arc_tolerance,
    int precision)
{
    PathsD result = Clipper2Lib::InflatePaths(
        to_clipper(paths),
        delta,
        to_join_type(join_type),
        to_end_type(end_type),
        miter_limit,
        precision,
        arc_tolerance);
    return from_clipper(result);
}

static Paths2 clip_lines(
    const Paths2 &lines,
    const Paths2 &boundary,
    int fill_rule,
    int precision)
{
    ClipperD clipper(precision);
    clipper.AddOpenSubject(to_clipper(lines));
    clipper.AddClip(to_clipper(boundary));

    PathsD closed_solution;  // not used: open subjects only ever appear in `open_solution`
    PathsD open_solution;
    if (!clipper.Execute(ClipType::Intersection, to_fill_rule(fill_rule), closed_solution, open_solution))
        throw std::runtime_error("Clipper2 failed to clip the lines against the boundary");

    return from_clipper(open_solution);
}

// --------------------------------------------------------------------------
// module
// --------------------------------------------------------------------------

NB_MODULE(_clipper2, m)
{
    m.doc() = "Minimal nanobind wrapper around Clipper2 (offsetting and open-path clipping).";

    m.def(
        "offset_paths",
        &offset_paths,
        "Offset (inflate / deflate) paths with Clipper2's InflatePaths.\n\n"
        "Parameters\n"
        "----------\n"
        "paths : list[list[[float, float]]]\n"
        "    Input paths, each a list of [x, y] points.\n"
        "delta : float\n"
        "    Offset distance. The sign relative to the path orientation decides\n"
        "    whether a closed polygon grows or shrinks.\n"
        "join_type : int\n"
        "    0=Square, 1=Bevel, 2=Round, 3=Miter.\n"
        "end_type : int\n"
        "    0=Polygon (closed), 1=Joined, 2=Butt, 3=Square, 4=Round.\n"
        "miter_limit : float\n"
        "    Miter limit, only used when join_type is Miter.\n"
        "arc_tolerance : float\n"
        "    Max deviation when approximating round joins/ends (0 = auto).\n"
        "precision : int\n"
        "    Decimal places Clipper2 keeps internally (0..8).\n\n"
        "Returns\n"
        "-------\n"
        "list[list[[float, float]]]\n"
        "    The offset paths.",
        "paths"_a,
        "delta"_a,
        "join_type"_a,
        "end_type"_a,
        "miter_limit"_a,
        "arc_tolerance"_a,
        "precision"_a);

    m.def(
        "clip_lines",
        &clip_lines,
        "Intersect open polylines with closed boundary polygons.\n\n"
        "Keeps only the portions of `lines` that fall inside `boundary`,\n"
        "returning them as open paths (segments).\n\n"
        "Parameters\n"
        "----------\n"
        "lines : list[list[[float, float]]]\n"
        "    Open polylines to clip, each a list of [x, y] points.\n"
        "boundary : list[list[[float, float]]]\n"
        "    Closed boundary polygons (the clip region), each a list of [x, y]\n"
        "    points. Multiple rings are combined using the fill rule.\n"
        "fill_rule : int\n"
        "    0=EvenOdd, 1=NonZero, 2=Positive, 3=Negative.\n"
        "precision : int\n"
        "    Decimal places Clipper2 keeps internally (0..8).\n\n"
        "Returns\n"
        "-------\n"
        "list[list[[float, float]]]\n"
        "    The clipped line segments that lie inside the boundary.",
        "lines"_a,
        "boundary"_a,
        "fill_rule"_a,
        "precision"_a);
}
