# Roadmap

## Vision
Use security camera video and local AI vision models to identify interesting events around my home in real time.  Store photos and video clips of these events.  Send alerts for particularly interesting events.

## Current Status
* The real-time program (`permon.monitor`) alerts reliably on US Mail, FedEx, UPS and Amazon trucks.
* The real-time program saves photos of these vehicles and others so that they can be used to further fine-tune the vision model.
* A web server program (`permon.viewer`) displays a gallery of the collected vehicle photos.
* Integration with FiftyOne (`permon.labeling`) for importing, reviewing, and labeling collected photos.

## Near-Term Focus
Refinements with a single camera – see [GitHub issues](https://github.com/awaldis/perimeter-monitor/issues).

## Medium-Term Plans
* Add more vehicle types to be detected (e.g. garbage truck, lawn service truck/trailer, emergency vehicles).
* Add automatic flagging of suspected false detections with simple heuristics (e.g. mail truck detected more than twice in a single day).
* Add automatic flagging of suspected false detections with a non-realtime, more sophisticated model double-checking the real-time model.
* Allow manual flagging of false detections via the web server.
* Expand the crop on the current camera so that more of the driveway can be seen.
* Add detection of pedestrians in the street and in the driveway.
* Capture video from the front door camera and detect people approaching the door.  (Open question: is additional hardware needed?)

## Future Possibilities
* Process more camera feeds that look for suspicious activity or interesting animals in the backyard.
* Expand beyond camera-based monitoring and alerts (e.g. temperature monitoring for refrigerators and freezers, furnace error codes).
* Decide whether to keep using this custom program or integrate with an off-the-shelf solution like Home Assistant.
