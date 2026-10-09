// The serving-overview concept module: re-exports the lesson and mounts it with the track's data (template rule 1).
import { mountLesson } from '@shared/lesson-page.js';
import { LESSON, lessonFor } from './serving-overview/content.js';

export { LESSON };
export default { slug: 'serving-overview', title: 'One request\'s journey', mount: (el, ctx) => mountLesson(el, ctx, lessonFor(ctx?.data)) };
