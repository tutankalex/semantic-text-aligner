from .schema import AlignmentCase, AlignmentIO

# see https://huggingface.co/datasets/vagrawal787/todo_task_list_types
SUBCASES = [
    AlignmentCase(
        name="mixed_task_subspan",
        description="Shorter span of the mixed list to keep comparisons fast.",
        input=AlignmentIO("""\
watch a tutorial to learn a new skill                                   , watch a tutorial
create a list of emergency contacts and distribute it to family members	, create a list of emergency contacts
share stories and memories with family members                          , distribute it to family members
set boundaries for personal time                                        , share stories and memories
"""),
        # TODO: update expected alignment once the algorithm is finalized.
        expected=AlignmentIO("""\
watch a tutorial to learn a new skill                                   , watch a tutorial
create a list of emergency contacts and distribute it to family members	, create a list of emergency contacts
                                                                        , distribute it to family members
share stories and memories with family members                          , share stories and memories
set boundaries for personal time                                        ,
"""),
    ),
    AlignmentCase(
        name="mixed_task_chunk4_ov2_span1",
        description="Chunk-size 4, overlap 2: second span of mixed task list.",
        input=AlignmentIO("""\
share stories and memories with family members                          , distribute it to family members
set boundaries for personal time                                        , share stories and memories
share and teach family recipes to the next generation                   , set boundaries for personal time
read a chapter from a book                                              , share and teach family recipes
"""),
        expected=AlignmentIO("""\
                                                                        , distribute it to family members
share stories and memories with family members                          , share stories and memories
set boundaries for personal time                                        , set boundaries for personal time
share and teach family recipes to the next generation                   , share and teach family recipes
read a chapter from a book                                              ,
"""),
    ),
    AlignmentCase(
        name="mixed_task_chunk4_ov2_span2",
        description="Chunk-size 4, overlap 2: third span of mixed task list.",
        input=AlignmentIO("""\
share and teach family recipes to the next generation                   , set boundaries for personal time
read a chapter from a book                                              , share and teach family recipes
plan and cook a healthy dinner                                          , read a chapter from a book
take out the trash and recycling                                        , plan and cook a healthy dinner
"""),
        expected=AlignmentIO("""\
                                                                        , set boundaries for personal time
share and teach family recipes to the next generation                   , share and teach family recipes
read a chapter from a book                                              , read a chapter from a book
plan and cook a healthy dinner                                          , plan and cook a healthy dinner
take out the trash and recycling                                        ,
"""),
    ),
    AlignmentCase(
        name="mixed_task_chunk4_ov2_span3",
        description="Chunk-size 4, overlap 2: fourth span of mixed task list.",
        input=AlignmentIO("""\
plan and cook a healthy dinner                                          , read a chapter from a book
take out the trash and recycling                                        , plan and cook a healthy dinner
schedule and conduct interviews for new hires                           , take out the trash
learn how to shut off gas and water supplies in your home               , recycling
"""),
        expected=AlignmentIO("""\
                                                                        , read a chapter from a book
plan and cook a healthy dinner                                          , plan and cook a healthy dinner
take out the trash and recycling                                        , take out the trash
                                                                        , recycling
schedule and conduct interviews for new hires                           ,
learn how to shut off gas and water supplies in your home               ,
"""),
    ),
    AlignmentCase(
        name="mixed_task_chunk3_ov1_span0",
        description="Chunk-size 3, overlap 1: first span of mixed task list.",
        input=AlignmentIO("""\
watch a tutorial to learn a new skill                                   , watch a tutorial
create a list of emergency contacts and distribute it to family members	, create a list of emergency contacts
share stories and memories with family members                          , distribute it to family members
"""),
        expected=AlignmentIO("""\
watch a tutorial to learn a new skill                                   , watch a tutorial
create a list of emergency contacts and distribute it to family members	, create a list of emergency contacts
                                                                        , distribute it to family members
share stories and memories with family members                          ,
"""),
    ),
    # index 5
    AlignmentCase(
        name="mixed_task_chunk3_ov1_span1",
        description="Chunk-size 3, overlap 1: second span of mixed task list.",
        input=AlignmentIO("""\
share stories and memories with family members                          , distribute it to family members
set boundaries for personal time                                        , share stories and memories
share and teach family recipes to the next generation                   , set boundaries for personal time
"""),
        expected=AlignmentIO("""\
                                                                        , distribute it to family members
share stories and memories with family members                          , share stories and memories
set boundaries for personal time                                        , set boundaries for personal time
share and teach family recipes to the next generation                   ,
"""),
    ),
    AlignmentCase(
        name="mixed_task_chunk3_ov1_span2",
        description="Chunk-size 3, overlap 1: third span of mixed task list.",
        input=AlignmentIO("""\
share and teach family recipes to the next generation                   , set boundaries for personal time
read a chapter from a book                                              , share and teach family recipes
plan and cook a healthy dinner                                          , read a chapter from a book
"""),
        expected=AlignmentIO("""\
                                                                        , set boundaries for personal time
share and teach family recipes to the next generation                   , share and teach family recipes
read a chapter from a book                                              , read a chapter from a book
plan and cook a healthy dinner                                          ,
"""),
    ),
    AlignmentCase(
        name="mixed_task_chunk7_ov3_span0",
        description="Chunk-size 7, overlap 3: first span of mixed task list.",
        input=AlignmentIO("""\
watch a tutorial to learn a new skill                                   , watch a tutorial
create a list of emergency contacts and distribute it to family members	, create a list of emergency contacts
share stories and memories with family members                          , distribute it to family members
set boundaries for personal time                                        , share stories and memories
share and teach family recipes to the next generation                   , set boundaries for personal time
read a chapter from a book                                              , share and teach family recipes
plan and cook a healthy dinner                                          , read a chapter from a book
"""),
        expected=AlignmentIO("""\
watch a tutorial to learn a new skill                                   , watch a tutorial
create a list of emergency contacts and distribute it to family members	, create a list of emergency contacts
                                                                        , distribute it to family members
share stories and memories with family members                          , share stories and memories
set boundaries for personal time                                        , set boundaries for personal time
share and teach family recipes to the next generation                   , share and teach family recipes
read a chapter from a book                                              , read a chapter from a book
plan and cook a healthy dinner                                          ,
"""),
    ),
    AlignmentCase(
        name="mixed_task_chunk7_ov3_span1",
        description="Chunk-size 7, overlap 3: second span of mixed task list.",
        input=AlignmentIO("""\
share and teach family recipes to the next generation                   , set boundaries for personal time
read a chapter from a book                                              , share and teach family recipes
plan and cook a healthy dinner                                          , read a chapter from a book
take out the trash and recycling                                        , plan and cook a healthy dinner
schedule and conduct interviews for new hires                           , take out the trash
learn how to shut off gas and water supplies in your home               , recycling
reach out to a friend you haven't spoken to in a while                  , schedule and
"""),
        expected=AlignmentIO("""\
                                                                        , set boundaries for personal time
share and teach family recipes to the next generation                   , share and teach family recipes
read a chapter from a book                                              , read a chapter from a book
plan and cook a healthy dinner                                          , plan and cook a healthy dinner
take out the trash and recycling                                        , take out the trash
                                                                        , recycling
schedule and conduct interviews for new hires                           , schedule and
learn how to shut off gas and water supplies in your home               ,
reach out to a friend you haven't spoken to in a while                  ,
"""),
    ),
    # index 9
    AlignmentCase(
        name="mixed_task_chunk7_ov3_span2",
        description="Chunk-size 7, overlap 3: third span of mixed task list.",
        input=AlignmentIO("""\
schedule and conduct interviews for new hires                          , take out the trash
learn how to shut off gas and water supplies in your home              , recycling
reach out to a friend you haven't spoken to in a while                 , schedule and
plan a virtual hangout with friends                                    , reach out to a friend you haven't spoken to in a while
take a digital detox for an hour                                       , plan a virtual hangout with friends
volunteer as a family for a local charity                              , volunteer as a family for a local charity
dust and clean ceiling fans and light fixtures                         , 
"""),
        expected=AlignmentIO("""\
                                                                        , take out the trash
                                                                        , recycling
schedule and conduct interviews for new hires                           , schedule and
learn how to shut off gas and water supplies in your home               ,
reach out to a friend you haven't spoken to in a while                  , reach out to a friend you haven't spoken to in a while
plan a virtual hangout with friends                                     , plan a virtual hangout with friends
take a digital detox for an hour                                        ,
volunteer as a family for a local charity                               , volunteer as a family for a local charity
dust and clean ceiling fans and light fixtures                          ,
"""),
    ),
]
