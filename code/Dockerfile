FROM localhost/hyst
RUN apt-get update && apt-get -qy install python-jinja2
ADD . /arch19-iotiming/
WORKDIR /arch19-iotiming/

WORKDIR /arch19-iotiming/
# # some simplified tests:
# # Run SpaceEx directly
# RUN /tmp/hyst/docker_entrypoint.sh spaceex -v l -g dummy/dummy.cfg -m dummy/dummy.xml
# # Run SpaceEx via hybridpy main function
# RUN /tmp/hyst/docker_entrypoint.sh python -m hybridpy.tool_spaceex dummy/dummy.xml - | fgrep -q "Tool script exit code: 0"

CMD bash
